"""
Custom evaluation script for real robot deployment.
Subscribes to ROS topics matching shape_meta keys and publishes actions via ROS publisher.
Simplified version: No TCP/Gripper separation, just unified action control.
"""
import pathlib
import threading
import time
import os
import os.path as osp
import numpy as np
import torch
import dill
import hydra
import cv2
import psutil
import signal
import sys
import gc
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Dict, Tuple, Union, Optional
from copy import deepcopy
import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, Header
from cv_bridge import CvBridge

from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.policy.base_image_policy import BaseImagePolicy
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.common.precise_sleep import precise_sleep
from reactive_diffusion_policy.common.ring_buffer import RingBuffer
from reactive_diffusion_policy.common.ensemble import EnsembleBuffer
from reactive_diffusion_policy.common.action_utils import (
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
)
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from loguru import logger

from reactive_diffusion_policy.model.common.pca_embedding import PCAEmbedding

PCA_DIR = "/home/yfx/turbo/reactive_diffusion_policy/data/PCA_Transform_GelSight"

# Set thread limits
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
cv2.setNumThreads(12)

# Set CPU affinity
total_cores = psutil.cpu_count()
num_cores_to_bind = 10
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
os.sched_setaffinity(0, cores_to_bind)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def stack_last_n_obs(all_obs, n_steps: int) -> np.ndarray:
    """Stack last n observations."""
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        result[:start_idx] = result[start_idx]
    return result

def apply_pca_to_marker(
    marker_flat: np.ndarray,
    pca_dir: str,
    n_components: int = 15,
) -> np.ndarray:
    """
    对 Marker_Tracking_Right_DXDY 的 (N, 126) 做 PCA 降维，返回 (N, n_components)。
    pca_dir 下需有 pca_transform_matrix.npy、pca_mean_matrix.npy（与 RDP McTac/GelSight 一致）。
    """
    pca_dir = os.path.expanduser(pca_dir)
    transform_path = os.path.join(pca_dir, "pca_transform_matrix.npy")
    mean_path = os.path.join(pca_dir, "pca_mean_matrix.npy")
    if not os.path.isfile(transform_path) or not os.path.isfile(mean_path):
        raise FileNotFoundError(
            f"PCA 目录需包含 pca_transform_matrix.npy 与 pca_mean_matrix.npy: {pca_dir}"
        )
    pca = PCAEmbedding(
        n_components=n_components,
        normalize=False,
        mode="Eval",
        store=False,
        transformation_matrix_path=transform_path,
        mean_matrix_path=mean_path,
    )
    # marker_flat: (N, 126)
    emb = pca.pca_reduction(marker_flat)  # (N, n_components)
    return emb.astype(np.float32)

class ROSObservationSubscriber:
    """ROS1 node that subscribes to observation topics matching shape_meta keys."""
    
    def __init__(self, shape_meta: DictConfig, topic_prefix: str = "/", 
                 topic_mapping: Optional[Dict[str, str]] = None,
                 force_subscribe_topics: Optional[Dict[str, str]] = None,
                 max_fps: int = 30, sync_slop: float = 0.4, sync_queue_size: int = 40):
        # Initialize ROS1 node if not already initialized
        if not rospy.get_node_uri():
            rospy.init_node('ros_obs_subscriber', anonymous=True)
        
        self.shape_meta = dict(shape_meta)
        self.topic_prefix = topic_prefix
        self.topic_mapping = topic_mapping or {}  # Maps shape_meta key to actual ROS topic name
        self.force_subscribe_topics = force_subscribe_topics or {}  # Force subscribe topics: {topic_name: msg_type}
        self.max_fps = max_fps
        self.bridge = CvBridge()
        
        # Parse shape_meta
        self.obs_keys = list(self.shape_meta.get('obs', {}).keys())
        self.extended_obs_keys = list(self.shape_meta.get('extended_obs', {}).keys()) if 'extended_obs' in self.shape_meta else []
        
        
        # Create subscribers
        self.subscribers = []
        self.topic_to_key = {}
        self.key_to_msg_type = {}
        print(self.obs_keys)
        # print(self.topic_mapping)
        # Subscribe to observation topics
        for key in self.obs_keys:
            obs_type = self.shape_meta['obs'][key].get('type', 'low_dim')
            # Use topic_mapping if available, otherwise use key directly
            if key in self.topic_mapping:
                topic_name = self.topic_mapping[key]
            #     # Ensure topic starts with /
            #     if not topic_name.startswith('/'):
            #         topic_name = f"/{topic_name}"
            # else:
            #     topic_name = f"{self.topic_prefix}{key}"
            
            # Check if there's explicit msg_type specified in shape_meta
            # if 'msg_type' in self.shape_meta['obs'][key]:
            #     msg_type_str = self.shape_meta['obs'][key]['msg_type']
            #     if msg_type_str == 'JointState':
            #         msg_type = JointState
            #         self.key_to_msg_type[key] = 'joint_state'
            #     elif msg_type_str == 'Image':
            #         msg_type = Image
            #         self.key_to_msg_type[key] = 'image'
            #     elif msg_type_str == 'Float32MultiArray':
            #         msg_type = Float32MultiArray
            #         self.key_to_msg_type[key] = 'float32_array'
            #     else:
            #         # Default based on obs_type
            #         if obs_type == 'rgb':
            #             msg_type = Image
            #             self.key_to_msg_type[key] = 'image'
            #         else:
            #             msg_type = Float32MultiArray
            #             self.key_to_msg_type[key] = 'float32_array'
            # else:
                # Auto-detect based on topic name or obs_type
                if 'joint' in key.lower() and 'state' in key.lower():
                    # Likely a JointState topic
                    msg_type = JointState
                    self.key_to_msg_type[key] = 'joint_state'
                elif obs_type == 'rgb':
                    msg_type = Image
                    self.key_to_msg_type[key] = 'image'
                else:
                    msg_type = Float32MultiArray
                    self.key_to_msg_type[key] = 'float32_array'
            else:
                msg_type = Float32MultiArray
                self.key_to_msg_type[key] = 'float32_array'
                topic_name = "/Marker_Tracking_Right_DXDY"
            # ROS1: message_filters.Subscriber directly takes topic name and message type
            sub = Subscriber(topic_name, msg_type)
            self.subscribers.append(sub)
            self.topic_to_key[topic_name] = key
            # print(self.topic_to_key)
            logger.info(f"Subscribed to topic: {topic_name} (key: {key}, msg_type: {self.key_to_msg_type[key]})")
            #obs successfully


        # Subscribe to extended observation topics
        for key in self.extended_obs_keys:
            obs_type = self.shape_meta['extended_obs'][key].get('type', 'low_dim')
            # Use topic_mapping if available, otherwise use key directly
            if key in self.topic_mapping:
                topic_name = self.topic_mapping[key]
                # Ensure topic starts with /
                # if not topic_name.startswith('/'):
                #     topic_name = f"/{topic_name}"
            else:
                topic_name = f"{self.topic_prefix}{key}"

            #topic name = /Marker_Tracking_Right_DXDY
            
            # Check if there's explicit msg_type specified in shape_meta
            # if 'msg_type' in self.shape_meta['extended_obs'][key]:
            #     msg_type_str = self.shape_meta['extended_obs'][key]['msg_type']
            #     if msg_type_str == 'JointState':
            #         msg_type = JointState
            #         self.key_to_msg_type[key] = 'joint_state'
            #     elif msg_type_str == 'Image':
            #         msg_type = Image
            #         self.key_to_msg_type[key] = 'image'
            #     elif msg_type_str == 'Float32MultiArray':
            #         msg_type = Float32MultiArray
            #         self.key_to_msg_type[key] = 'float32_array'
            #     else:
            #         if obs_type == 'rgb':
            #             msg_type = Image
            #             self.key_to_msg_type[key] = 'image'
            #         else:
            #             msg_type = Float32MultiArray
            #             self.key_to_msg_type[key] = 'float32_array'

                # Auto-detect based on topic name or obs_type
            if 'joint' in key.lower() and 'state' in key.lower():
                msg_type = JointState
                self.key_to_msg_type[key] = 'joint_state'
            elif obs_type == 'rgb':
                msg_type = Image
                self.key_to_msg_type[key] = 'image'
            else:
                msg_type = Float32MultiArray
                self.key_to_msg_type[key] = 'float32_array'
            ###msg_type = float32_array
            topic_name = "/Marker_Tracking_Right_DXDY"
            sub = Subscriber(topic_name, msg_type)
            self.subscribers.append(sub)
            self.topic_to_key[topic_name] = key
            logger.info(f"Subscribed to extended obs topic: {topic_name} (key: {key}, msg_type: {self.key_to_msg_type[key]})")

        # Create time synchronizer
        if len(self.subscribers) > 0:
            self.ts = ApproximateTimeSynchronizer(
                self.subscribers, 
                queue_size=sync_queue_size, 
                slop=sync_slop,
                allow_headerless=True
            )
            self.ts.registerCallback(self.callback)
        
        # Observation buffer
        self.obs_buffer = RingBuffer(size=1024, fps=max_fps)
        self.mutex = threading.Lock()
    
    def callback(self, *msgs):
        """Callback for synchronized messages."""
        topic_dict = {}
        for i, msg in enumerate(msgs):
            # ROS1: Get topic name from subscriber's topic attribute
            if hasattr(self.subscribers[i], 'topic'):
                topic_name = self.subscribers[i].topic
            else:
                # Fallback: use topic name from shape_meta
                if i < len(self.obs_keys):
                    key = self.obs_keys[i]
                elif i < len(self.obs_keys) + len(self.extended_obs_keys):
                    key = self.extended_obs_keys[i - len(self.obs_keys)]
                else:
                    continue
                topic_name = f"{self.topic_prefix}{key}"
            topic_dict[topic_name] = msg
        
        # Convert messages to numpy arrays
        obs_dict = {}
        timestamp = None
        
        for topic_name, msg in topic_dict.items():
            key = self.topic_to_key[topic_name]
            msg_type = self.key_to_msg_type[key]
            
            # Extract timestamp (ROS1 uses secs/nsecs, ROS2 uses sec/nanosec)
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                if hasattr(msg.header.stamp, 'secs'):  # ROS1
                    msg_timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
                else:  # ROS2 (fallback)
                    msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if timestamp is None:
                    timestamp = msg_timestamp
                else:
                    timestamp = max(timestamp, msg_timestamp)
            
            # Convert message to numpy array
            if msg_type == 'image':
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    # Assume image is already RGB format, no BGR2RGB conversion needed
                    rgb_image = cv_image
                    
                    # Get expected shape
                    if key in self.obs_keys:
                        expected_shape = self.shape_meta['obs'][key]['shape']
                    elif key in self.extended_obs_keys:
                        expected_shape = self.shape_meta['extended_obs'][key]['shape']
                    else:
                        expected_shape = None
                    
                    # Keep HWC format for get_real_obs_dict() which expects THWC
                    # get_real_obs_dict() will handle resize and THWC to TCHW conversion
                    if expected_shape and len(expected_shape) == 3:  # (C, H, W)
                        _, h, w = expected_shape
                        if rgb_image.shape[:2] != (h, w):
                            rgb_image = cv2.resize(rgb_image, (w, h))
                    # Don't convert to CHW here - keep HWC format
                    # get_real_obs_dict() expects THWC and will convert to TCHW
                    
                    obs_dict[key] = rgb_image
                except Exception as e:
                    logger.error(f"Failed to convert image for {key}: {e}")
                    continue
                    
            elif msg_type == 'joint_state':
                # Extract position from JointState message
                try:
                    if hasattr(msg, 'position') and len(msg.position) > 0:
                        data = np.array(msg.position, dtype=np.float32)
                    else:
                        logger.warning(f"JointState message for {key} has no position data")
                        continue
                    
                    # Get expected shape
                    if key in self.obs_keys:
                        expected_shape = self.shape_meta['obs'][key]['shape']
                    elif key in self.extended_obs_keys:
                        expected_shape = self.shape_meta['extended_obs'][key]['shape']
                    else:
                        expected_shape = None
                    
                    if expected_shape and len(expected_shape) == 1:
                        if data.size != expected_shape[0]:
                            print("error: the joint state is not good!!")
                            if data.size > expected_shape[0]:
                                data = data[:expected_shape[0]]
                            else:
                                padded = np.zeros(expected_shape[0], dtype=np.float32)
                                padded[:data.size] = data
                                data = padded
                    obs_dict[key] = data
                except Exception as e:
                    logger.error(f"Failed to convert JointState for {key}: {e}")
                    continue
                    
            elif msg_type == 'float32_array':
                if hasattr(msg, 'data'):
                    data = np.array(msg.data, dtype=np.float32)
                    
                    # Get expected shape
                    if key in self.obs_keys:
                        expected_shape = self.shape_meta['obs'][key]['shape']
                    elif key in self.extended_obs_keys:
                        expected_shape = self.shape_meta['extended_obs'][key]['shape']
                    else:
                        expected_shape = None
                    expected_shape = [126]
                    if expected_shape and len(expected_shape) == 1:
                        if data.size != expected_shape[0]:
                            if data.size > expected_shape[0]:
                                data = data[:expected_shape[0]]
                            else:
                                padded = np.zeros(expected_shape[0], dtype=np.float32)
                                padded[:data.size] = data
                                data = padded
                    ###To do  (make embedding)
                    data = apply_pca_to_marker(data,pca_dir=PCA_DIR,n_components=15)
                    obs_dict[key] = data
        
        # Add timestamp
        if timestamp is not None:
            obs_dict['timestamp'] = np.array([timestamp])
        
        # Push to buffer
        if len(obs_dict) > 0:
            self.obs_buffer.push(obs_dict)
    
    def get_obs(self, obs_steps: int = 2, temporal_downsample_ratio: int = 2) -> Dict[str, np.ndarray]:
        """Get observations with temporal downsampling."""
        last_n_obs_list, _ = self.obs_buffer.peek_last_n(obs_steps * temporal_downsample_ratio)
        
        result = {}
        last_n_obs_list = [obs for obs in last_n_obs_list if obs is not None]
        if len(last_n_obs_list) == 0:
            return result
        
        # Apply temporal downsampling
        downsampled_obs_list = last_n_obs_list[::temporal_downsample_ratio]
        downsampled_obs_list = downsampled_obs_list[:obs_steps]
        downsampled_obs_list = downsampled_obs_list[::-1]  # oldest to newest
        
        # Stack observations
        for key in downsampled_obs_list[0].keys():
            if key == 'timestamp':
                timestamps = [obs[key] for obs in downsampled_obs_list]
                result[key] = np.array(timestamps)
            else:
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in downsampled_obs_list], 
                    obs_steps
                )
        
        return result
    
    def reset(self):
        """Reset observation buffer."""
        self.obs_buffer.reset()



class ActionPublisher:
    """ROS1 node that publishes actions as JointState messages."""
    
    def __init__(self, action_topic: str = "/right_arm/joint_ctrl_single", 
                 joint_names: Optional[list] = None):
        # Initialize ROS1 node if not already initialized
        if not rospy.get_node_uri():
            rospy.init_node('action_publisher', anonymous=True)
        self.action_publisher = rospy.Publisher(action_topic, JointState, queue_size=10)
        self.joint_names = joint_names  # Will be set from shape_meta if None
        logger.info(f"Created action publisher on topic: {action_topic}")
        if self.joint_names:
            logger.info(f"Joint names: {self.joint_names}")
    
    def set_joint_names(self, joint_names: list):
        """Set joint names from shape_meta or configuration."""
        self.joint_names = joint_names
        logger.info(f"Joint names set to: {self.joint_names}")
    
    def publish_action(self, action: np.ndarray):
        """Publish action to ROS topic as JointState message."""
        action = action.flatten()
        action_dim = action.shape[0]
        
        # Create JointState message
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        
        # Set joint names (use provided names or generate default names)
        if self.joint_names is not None:
            if len(self.joint_names) == action_dim:
                joint_state_msg.name = self.joint_names
            else:
                logger.warning(f"Joint names length ({len(self.joint_names)}) doesn't match action dim ({action_dim}). Using default names.")
                joint_state_msg.name = [f'joint_{i}' for i in range(action_dim)]
        else:
            # Generate default joint names
            joint_state_msg.name = [f'joint_{i}' for i in range(action_dim)]
        
        # Set joint positions (actions are joint positions/commands)
        joint_state_msg.position = action.tolist()
        
        # Optionally set velocity and effort to empty (or zeros)
        joint_state_msg.velocity = []
        joint_state_msg.effort = []
        
        self.action_publisher.publish(joint_state_msg)
        logger.debug(f"Published action (JointState) with {action_dim} joints: {action}")


class CustomRealRunner:
    """Custom real runner - simplified: no TCP/Gripper separation, unified action control."""
    
    def __init__(self,
                 shape_meta: DictConfig,
                 transform_params: DictConfig,
                 topic_prefix: str = "/",
                 topic_mapping: Optional[Dict[str, str]] = None,
                 force_subscribe_topics: Optional[Dict[str, str]] = None,
                 action_topic: str = "/right_arm/joint_ctrl_single",
                 joint_names: Optional[list] = None,
                 action_ensemble_buffer_params: DictConfig = None,  # Single buffer for all actions
                 latent_action_ensemble_buffer_params: DictConfig = None,
                 use_latent_action_with_rnn_decoder: bool = False,
                 use_relative_action: bool = False,
                 action_interpolation_ratio: int = 1,
                 eval_episodes: int = 10,
                 max_duration_time: float = 30,
                 action_update_interval: int = 6,  # Single update interval
                 action_clip_range: ListConfig = None,  # Clip range for actions
                 control_fps: float = 24,
                 inference_fps: float = 6,
                 latency_step: int = 0,
                 n_obs_steps: int = 2,
                 obs_temporal_downsample_ratio: int = 2,
                 dataset_obs_temporal_downsample_ratio: int = 1,
                 downsample_extended_obs: bool = True,
                 task_name: str = None):
        
        self.task_name = task_name
        self.transforms = RealWorldTransforms(option=transform_params)
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes
        
        # Parse shape_meta
        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                rgb_keys.append(key)
            elif obs_type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        
        extended_rgb_keys = []
        extended_lowdim_keys = []
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                extended_rgb_keys.append(key)
            elif obs_type == 'low_dim':
                extended_lowdim_keys.append(key)
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys
        
        # Control parameters
        self.max_duration_time = max_duration_time
        self.action_update_interval = action_update_interval
        self.action_clip_range = action_clip_range
        self.control_fps = control_fps
        self.control_interval_time = 1.0 / control_fps
        self.inference_fps = inference_fps
        self.inference_interval_time = 1.0 / inference_fps
        assert self.control_fps % self.inference_fps == 0
        self.latency_step = latency_step
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self.downsample_extended_obs = downsample_extended_obs
        self.use_latent_action_with_rnn_decoder = use_latent_action_with_rnn_decoder
        self.use_relative_action = use_relative_action
        self.action_interpolation_ratio = action_interpolation_ratio
        
        # Single ensemble buffer for all actions (no TCP/Gripper separation)
        if self.use_latent_action_with_rnn_decoder:
            assert latent_action_ensemble_buffer_params.ensemble_mode == 'new'
            self.action_ensemble_buffer = EnsembleBuffer(**latent_action_ensemble_buffer_params)
        else:
            self.action_ensemble_buffer = EnsembleBuffer(**action_ensemble_buffer_params)
        
        # Initialize ROS1 nodes (rospy.init_node is called inside the classes)
        self.obs_subscriber = ROSObservationSubscriber(
            shape_meta=shape_meta,
            topic_prefix=topic_prefix,
            topic_mapping=topic_mapping,
            force_subscribe_topics=force_subscribe_topics,
            max_fps=30,
            sync_slop=0.4,
            sync_queue_size=40
        )
        self.action_publisher_node = ActionPublisher(action_topic=action_topic, joint_names=joint_names)
        
        # Set joint names if available from shape_meta
        if joint_names is None:
            # Try to extract joint names from shape_meta if available
            action_shape = self.shape_meta.get('action', {}).get('shape', [])
            if isinstance(action_shape, (list, tuple)) and len(action_shape) > 0:
                action_dim = action_shape[0]
                # Check if there's joint name info in config
                if 'joint_names' in self.shape_meta.get('action', {}):
                    joint_names_from_meta = self.shape_meta['action']['joint_names']
                    if isinstance(joint_names_from_meta, (list, tuple)) and len(joint_names_from_meta) == action_dim:
                        self.action_publisher_node.set_joint_names(list(joint_names_from_meta))
        
        self.stop_event = threading.Event()
        self.action_step_count = 0
        self.shutdown_requested = False
    
    def _cleanup_memory(self):
        """Clean up memory and free resources."""
        logger.info("Starting memory cleanup...")
        
        # Clear ensemble buffers
        if hasattr(self, 'action_ensemble_buffer'):
            self.action_ensemble_buffer.clear()
            del self.action_ensemble_buffer
        
        # Clear observation subscriber buffers
        if hasattr(self, 'obs_subscriber') and hasattr(self.obs_subscriber, 'obs_buffer'):
            self.obs_subscriber.obs_buffer.reset()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")
        
        # Force garbage collection multiple times
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                logger.debug(f"Garbage collection cycle {i+1}: freed {collected} objects")
        
        # Clear large numpy arrays if any
        import sys
        for name in list(sys.modules.keys()):
            if 'numpy' in name or 'cv2' in name:
                # Don't delete these, just clear any cached data
                pass
        
        # Log memory usage
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.info(f"Memory after cleanup - RSS: {mem_info.rss / 1024 / 1024:.2f} MB, "
                       f"VMS: {mem_info.vms / 1024 / 1024:.2f} MB")
        except Exception as e:
            logger.debug(f"Could not get memory info: {e}")
        
        logger.info("Memory cleanup completed")
    
    def pre_process_extended_obs(self, extended_obs_dict: Dict) -> Tuple[Dict, Dict]:
        """Preprocess extended observations."""
        extended_obs_dict = deepcopy(extended_obs_dict)
        
        absolute_extended_obs_dict = {}
        for key in self.extended_lowdim_keys:
            extended_obs_dict[key] = extended_obs_dict[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]]
            absolute_extended_obs_dict[key] = extended_obs_dict[key].copy()
        
        if self.use_relative_action:
            for key in self.extended_lowdim_keys:
                if 'joint' in key.lower() or 'robot' in key.lower():
                    base_absolute_action = extended_obs_dict[key][-1].copy()
                    extended_obs_dict[key] = absolute_actions_to_relative_actions(
                        extended_obs_dict[key], 
                        base_absolute_action=base_absolute_action
                    )
        
        return extended_obs_dict, absolute_extended_obs_dict
    
    def post_process_action(self, action: np.ndarray) -> np.ndarray:
        """Post-process action: clip to limits if provided."""
        assert len(action.shape) == 2  # (action_steps, action_dim)
        
        # Clip action if clip range is provided
        if self.action_clip_range is not None:
            action = np.clip(
                action,
                np.array(self.action_clip_range[0]),
                np.array(self.action_clip_range[1])
            )
        return action
    
    def action_command_thread(self, policy: DiffusionUnetImagePolicy, stop_event):
        """Action execution thread - unified action control."""
        try:
            while not stop_event.is_set() and not self.shutdown_requested:
                start_time = time.time()
                # print("get latent action from ensemble buffer") 
                # Get action from ensemble buffer
                step_action = self.action_ensemble_buffer.get_action()
                if step_action is None:
                    cur_time = time.time()
                    sleep_time = max(0., self.control_interval_time - (cur_time - start_time))
                    # Check for shutdown during sleep
                    if sleep_time > 0:
                        sleep_interval = 0.1  # Check every 100ms
                        elapsed = 0
                        while elapsed < sleep_time and not self.shutdown_requested:
                            time.sleep(min(sleep_interval, sleep_time - elapsed))
                            elapsed += sleep_interval
                    else:
                        precise_sleep(sleep_time)
                    if self.shutdown_requested:
                        break
                    self.action_step_count += 1


                    continue
                
                if self.use_latent_action_with_rnn_decoder:
                    # Handle latent action with RNN decoder

                    ###########step_action 57,why?? it supposed 56?#############and also what is extended_obs_step
                    ####################################need to know
                    extended_obs_step = int(step_action[-1])
                    # extended_obs_step = 32
                    step_action = step_action[:-1]
                    # 注意：get_action 会修改 buffer（popleft），所以这里记录的是调用后的状态
                    # 但 timestep 在 get_action 内部会增加，所以这里显示的是增加后的值
    #                 print(f"Got action from buffer: extended_obs_step={extended_obs_step}, "
    #   f"buffer_timestep={self.action_ensemble_buffer.timestep}, "
    #   f"buffer_len={len(self.action_ensemble_buffer.actions)}, "
    #   f"actions_start_timestep={self.action_ensemble_buffer.actions_start_timestep}")
                    obs_temporal_downsample_ratio = self.obs_temporal_downsample_ratio if self.downsample_extended_obs else 1
                    extended_obs = self.obs_subscriber.get_obs(extended_obs_step, obs_temporal_downsample_ratio)
                    
                    if self.use_relative_action:
                        # Extract base absolute action (current state)
                        action_dim = self.shape_meta['action']['shape'][0]
                        base_absolute_action = step_action[-action_dim:]
                        step_action = step_action[:-action_dim]
     

                    np_extended_obs_dict = dict(extended_obs)
                    np_extended_obs_dict = get_real_obs_dict(
                        env_obs=np_extended_obs_dict,
                        shape_meta=self.shape_meta,
                        is_extended_obs=True
                    )

                    np_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict)

                    extended_obs_dict = dict_apply(
                        np_extended_obs_dict, 
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device=policy.device)
                    )
                    
                    step_latent_action = torch.from_numpy(step_action.astype(np.float32)).unsqueeze(0).to(device=policy.device)
                    step_action = policy.predict_from_latent_action(
                        step_latent_action, extended_obs_dict, extended_obs_step,
                        self.dataset_obs_temporal_downsample_ratio
                    )['action'][0].detach().cpu().numpy()
                    # Clear GPU tensors
                    del step_latent_action, extended_obs_dict
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if self.use_relative_action:
                        step_action = relative_actions_to_absolute_actions(
                            step_action, 
                            base_absolute_action
                        )
                    print(step_action.shape)
                    step_action = step_action[-1]  # Take last step 为什么是最后一步呢？
                else:
                    # Standard action (not latent)
                    step_action = step_action[-1]  # Take last step from buffer
                
                # Post-process and publish
                # for i in range(step_action.shape[0]//2):
                #     print(step_action.shape)
                #     step_action_single = step_action[i]
                #     step_action_single = self.post_process_action(step_action_single[np.newaxis, :])
                #     step_action_single = step_action_single.squeeze(0)
                #     self.action_publisher_node.publish_action(step_action_single)
                ################################process action "clip+smoothed"
                # Publish action via ROS
                step_action = self.post_process_action(step_action[np.newaxis, :])
                step_action = step_action.squeeze(0)
                self.action_publisher_node.publish_action(step_action)
                cur_time = time.time()
                sleep_time = max(0., self.control_interval_time - (cur_time - start_time))
                # Check for shutdown during sleep
                if sleep_time > 0:
                    sleep_interval = 0.1  # Check every 100ms
                    elapsed = 0
                    while elapsed < sleep_time and not self.shutdown_requested:
                        time.sleep(min(sleep_interval, sleep_time - elapsed))
                        elapsed += sleep_interval
                else:
                    precise_sleep(sleep_time)
                if self.shutdown_requested:
                    break
                self.action_step_count += 1
                
        except KeyboardInterrupt:
            logger.info("[ActionThread] KeyboardInterrupt received. Stopping...")
        except Exception as e:
            logger.error(f"[ActionThread] Error: {e}")
            raise
        finally:
            logger.info("[ActionThread] Exiting action command thread.")
            # Clean up any local variables
            try:
                if 'step_action' in locals():
                    del step_action
                if 'extended_obs_dict' in locals():
                    del extended_obs_dict
                if 'step_latent_action' in locals():
                    del step_latent_action
                if 'extended_obs' in locals():
                    del extended_obs
                gc.collect()
            except Exception:
                pass  # Ignore errors during cleanup
    
    def run(self, policy: DiffusionUnetImagePolicy):
        """Main run loop."""
        if self.use_latent_action_with_rnn_decoder:
            assert policy.at.use_rnn_decoder, "Policy should use rnn decoder for latent action."
        else:
            assert not hasattr(policy, 'at') or not policy.at.use_rnn_decoder, "Policy should not use rnn decoder for action."
        
        device = policy.device
        
        # ROS1: Start rospy.spin in a separate thread
        def spin_ros():
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                logger.info("ROS spin interrupted")
        
        spin_thread = threading.Thread(target=spin_ros, daemon=True)
        spin_thread.start()
        
        time.sleep(2)  # Wait for subscribers to connect
        
        try:
            for episode_idx in range(self.eval_episodes):
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping evaluation")
                    break
                    
                logger.info(f"Start evaluation episode {episode_idx}")
                
                # Reset
                self.obs_subscriber.reset()
                policy.reset()
                self.action_ensemble_buffer.clear()
                
                # Clear GPU cache and run garbage collection between episodes
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                self.stop_event.clear()
                time.sleep(0.5)
                
                # Start action command thread
                action_thread = threading.Thread(
                    target=self.action_command_thread,
                    args=(policy, self.stop_event),
                    daemon=True
                )
                action_thread.start()
                
                self.action_step_count = 0
                step_count = 0
                steps_per_inference = int(self.control_fps / self.inference_fps)
                start_timestamp = time.time()
                
                try:
                    while not self.shutdown_requested:
                        start_time = time.time()

                        # Get observations
                        obs = self.obs_subscriber.get_obs(
                            obs_steps=self.n_obs_steps,
                            temporal_downsample_ratio=self.obs_temporal_downsample_ratio
                        )
# camera_f                       | (2, 240, 320, 3)     | uint8
# camera_r                       | (2, 240, 320, 3)     | uint8
# joint_states_single            | (2, 7)               | float32
# Marker_Tracking_Right_DXDY_emb | (2, 15)              | float32  
# timestamp                      | (2, 1)               | float64


                        if len(obs) == 0:
                            logger.warning("No observation received! Skip this step.")
                            cur_time = time.time()
                            sleep_time = max(0., self.inference_interval_time - (cur_time - start_time))
                            # Check for shutdown during sleep
                            if sleep_time > 0:
                                sleep_interval = 0.1  # Check every 100ms
                                elapsed = 0
                                while elapsed < sleep_time and not self.shutdown_requested:
                                    time.sleep(min(sleep_interval, sleep_time - elapsed))
                                    elapsed += sleep_interval
                            else:
                                precise_sleep(sleep_time)
                            if self.shutdown_requested:
                                break
                            step_count += steps_per_inference
                            continue
                        
                        # Preprocess observations
                        np_obs_dict = dict(obs)
                        np_obs_dict = get_real_obs_dict(env_obs=np_obs_dict, shape_meta=self.shape_meta)



                        # Convert to torch
                        obs_dict = dict_apply(
                            np_obs_dict,
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device)
                        )
                        # Clear numpy dict to free memory
                        del np_obs_dict
                        # Run policy inference
                        policy_time = time.time()
                        #######################To do: Make the obs_dict input fit####################
                        with torch.no_grad():
                            if self.use_latent_action_with_rnn_decoder:
                                # print(obs_dict.keys())
                                action_dict = policy.predict_action(
                                    obs_dict,
                                    dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                    return_latent_action=True
                                )
                            else:
                                action_dict = policy.predict_action(obs_dict)

                        logger.debug(f"Policy inference time: {time.time() - policy_time:.3f}s")
                      # Convert to numpy and clear GPU memory
                        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                        # Clear GPU cache after moving to CPU
                        del action_dict
                        action_all = np_action_dict['action'].squeeze(0)  # (horizon, action_dim)
                        #action_all. shape = (29 , 56)  [To-1:Tend] To align with the obs
                        if self.use_latent_action_with_rnn_decoder:
                            
                            # Add action step for extended obs
                            action_all = np.concatenate([
                                action_all,
                                np.arange(
                                    self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio,
                                    action_all.shape[0] + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio
                                )[:, np.newaxis]
                            ], axis=-1)   
                            # action_all (29,57) 
                            # print(action_all.shape)
                            # print("ACTION NUMBER",action_all[:,-1])  4~32
                        if self.action_interpolation_ratio > 1:
                            if self.use_latent_action_with_rnn_decoder:
                                action_all = action_all.repeat(self.action_interpolation_ratio, axis=0)
                            else:
                                from reactive_diffusion_policy.common.action_utils import interpolate_actions_with_ratio
                                action_all = interpolate_actions_with_ratio(action_all, self.action_interpolation_ratio)
                        
                        # Add action to ensemble buffer
                        if step_count % self.action_update_interval == 0:
                            print("add laten action")
                            print(step_count, self.action_update_interval)
                            if self.use_latent_action_with_rnn_decoder:
                                action = action_all[self.latency_step:, ...]
                            else:
                                action = action_all[self.latency_step:, :]
                            #action.shape(25,57)
                            # 记录添加前后的 buffer 状态
                            # before_len = len(self.action_ensemble_buffer.actions)
                            # before_start = self.action_ensemble_buffer.actions_start_timestep
                            # before_timestep = self.action_ensemble_buffer.timestep
                            # idx = step_count - before_start
                            # print(f"[BEFORE ADD] buffer_len={before_len}, actions_start_timestep={before_start}, "
                            #       f"buffer_timestep={before_timestep}, idx={idx}, "
                            #       f"need_extend_to={idx + action.shape[0] - 1}")
                            # self.action_ensemble_buffer.clear()
                            self.action_ensemble_buffer.add_action(action, step_count)
                            # after_len = len(self.action_ensemble_buffer.actions)
                            # after_start = self.action_ensemble_buffer.actions_start_timestep
                            # print(f"[AFTER ADD] buffer_len={after_len}, actions_start_timestep={after_start}, "
                            #       f"buffer_span=[{after_start}, {after_start + after_len - 1 if after_len > 0 else after_start}], "
                            #       f"action_covers=[{step_count}, {step_count + action.shape[0] - 1}]")
                            # Clear action to free memory
                            del action
                        
                        # Clear intermediate variables
                        del action_all, np_action_dict
                        
                        # Periodic garbage collection every 100 steps
                        if step_count % (steps_per_inference * 100) == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        cur_time = time.time()
                        sleep_time = max(0., self.inference_interval_time - (cur_time - start_time))
                        # Check for shutdown during sleep
                        if sleep_time > 0:
                            sleep_interval = 0.1  # Check every 100ms
                            elapsed = 0
                            while elapsed < sleep_time and not self.shutdown_requested:
                                time.sleep(min(sleep_interval, sleep_time - elapsed))
                                elapsed += sleep_interval
                        else:
                            precise_sleep(sleep_time)
                        
                        if self.shutdown_requested:
                            logger.info("Shutdown requested during inference loop")
                            break
                            
                        if cur_time - start_timestamp >= self.max_duration_time:
                            logger.info(f"Episode {episode_idx} reaches max duration time {self.max_duration_time} seconds.")
                            break
                        step_count += steps_per_inference
                
                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt! Terminate the episode now!")
                    self.shutdown_requested = True
                finally:
                    self.stop_event.set()
                    logger.info("Waiting for action thread to finish...")
                    action_thread.join(timeout=2.0)
                    if action_thread.is_alive():
                        logger.warning("Action thread did not finish in time, continuing...")
            
            # Shutdown ROS spin thread
            logger.info("Shutting down ROS...")
            if not rospy.is_shutdown():
                rospy.signal_shutdown("Evaluation finished")
            spin_thread.join(timeout=1.0)
            if spin_thread.is_alive():
                logger.warning("ROS spin thread did not finish in time, continuing...")
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received in main loop!")
            self.shutdown_requested = True
            self.stop_event.set()
        except Exception as e:
            logger.error(f"Error in run loop: {e}")
            self.shutdown_requested = True
            self.stop_event.set()
            raise
        finally:
            # ROS1: Shutdown rospy
            logger.info("Cleaning up and freeing memory...")
            self._cleanup_memory()
            if not rospy.is_shutdown():
                rospy.signal_shutdown("Evaluation finished")
            self.stop_event.set()
            
# Global variable to store runner instance for signal handler
_runner_instance = None

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) signal."""
    global _runner_instance
    logger.warning("\nSIGINT (Ctrl+C) received! Initiating graceful shutdown...")
    if _runner_instance is not None:
        _runner_instance.shutdown_requested = True
        _runner_instance.stop_event.set()
    # Raise KeyboardInterrupt to propagate to main execution
    raise KeyboardInterrupt

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('reactive_diffusion_policy', 'config')),
    config_name="train_latent_diffusion_unet_real_image_workspace"
)
def main(cfg):
    global _runner_instance
    
    # Register signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load checkpoint
    ckpt_path = cfg.ckpt_path
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    
    # Initialize workspace and load policy
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Setup policy
    if 'diffusion' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        
        if 'latent' in cfg.name:
            policy.at.set_normalizer(policy.normalizer)
        
        device = torch.device('cuda')
        policy.eval().to(device)
        policy.num_inference_steps = 8
    else:
        raise NotImplementedError
    
    # Get config from task config
    task_cfg = cfg.task
    
    # Create custom runner - simplified: single action buffer (no TCP/Gripper separation)
    # If config has tcp_ensemble_buffer_params, use it; otherwise create default
    env_runner_cfg = task_cfg.env_runner
    if 'action_ensemble_buffer_params' in env_runner_cfg:
        action_ensemble_buffer_params = env_runner_cfg.action_ensemble_buffer_params
    elif 'tcp_ensemble_buffer_params' in env_runner_cfg:
        # Fallback: use tcp_ensemble_buffer_params if action_ensemble_buffer_params doesn't exist
        action_ensemble_buffer_params = env_runner_cfg.tcp_ensemble_buffer_params
    else:
        action_ensemble_buffer_params = DictConfig({'ensemble_mode': 'new'})
    
    if 'latent_action_ensemble_buffer_params' in env_runner_cfg:
        latent_action_ensemble_buffer_params = env_runner_cfg.latent_action_ensemble_buffer_params
    elif 'latent_tcp_ensemble_buffer_params' in env_runner_cfg:
        latent_action_ensemble_buffer_params = env_runner_cfg.latent_tcp_ensemble_buffer_params
    else:
        latent_action_ensemble_buffer_params = DictConfig({'ensemble_mode': 'new'})
    
    # Get action update interval (use tcp_action_update_interval as fallback)
    action_update_interval = env_runner_cfg.get('action_update_interval', 
        env_runner_cfg.get('tcp_action_update_interval', 6))
    
    # Get action clip range (optional)
    action_clip_range = env_runner_cfg.get('action_clip_range', None)
    
    # Get joint names from config if available
    joint_names = env_runner_cfg.get('joint_names', None)
    
    # Get topic mapping from config if available (maps shape_meta keys to actual ROS topic names)
    topic_mapping = env_runner_cfg.get('topic_mapping', None)
    if topic_mapping is None:
        # Default mapping based on common patterns - you can override this in config
        topic_mapping = {}
        # Auto-generate mapping for common cases
        for key in task_cfg.shape_meta.get('obs', {}).keys():
            if key == 'camera_f':
                topic_mapping[key] = '/camera_f/color/image_raw'
            elif key == 'camera_r':
                topic_mapping[key] = '/camera_r/color/image_raw'
            elif 'joint' in key.lower() and 'state' in key.lower():
                # For joint states, try common patterns
                if 'right_arm' not in key:
                    topic_mapping[key] = f'/right_arm/{key}'
    
    # Get force subscribe topics from config (topics to subscribe even if not in shape_meta)
    force_subscribe_topics = env_runner_cfg.get('force_subscribe_topics', None)
    if force_subscribe_topics is None:
        # Default: Always subscribe to Marker_Tracking_Right_DXDY
        force_subscribe_topics = {
            '/Marker_Tracking_Right_DXDY': 'Float32MultiArray'
        }
    else:
        # Ensure Marker_Tracking_Right_DXDY is always included
        if '/Marker_Tracking_Right_DXDY' not in force_subscribe_topics:
            force_subscribe_topics['/Marker_Tracking_Right_DXDY'] = 'Float32MultiArray'
    
    global _runner_instance
    runner = CustomRealRunner(
        shape_meta=task_cfg.shape_meta,
        transform_params=task_cfg.transforms,
        topic_prefix="/",  # Customize as needed
        topic_mapping=topic_mapping,  # Maps shape_meta keys to actual ROS topics
        force_subscribe_topics=force_subscribe_topics,  # Force subscribe topics (even if not in shape_meta)
        action_topic="/right_arm/joint_ctrl_single",  # Your action topic
        joint_names=joint_names,  # Joint names from config
        action_ensemble_buffer_params=action_ensemble_buffer_params,
        latent_action_ensemble_buffer_params=latent_action_ensemble_buffer_params,
        use_latent_action_with_rnn_decoder=env_runner_cfg.use_latent_action_with_rnn_decoder,
        use_relative_action=task_cfg.dataset.relative_action,
        eval_episodes=env_runner_cfg.eval_episodes,
        max_duration_time=env_runner_cfg.max_duration_time,
        action_update_interval=action_update_interval,
        action_clip_range=action_clip_range,
        control_fps=env_runner_cfg.control_fps,
        inference_fps=env_runner_cfg.inference_fps,
        latency_step=env_runner_cfg.latency_step,
        n_obs_steps=env_runner_cfg.n_obs_steps,
        obs_temporal_downsample_ratio=env_runner_cfg.obs_temporal_downsample_ratio,
        dataset_obs_temporal_downsample_ratio=env_runner_cfg.dataset_obs_temporal_downsample_ratio,
        downsample_extended_obs=env_runner_cfg.downsample_extended_obs,
        task_name=task_cfg.name
    )
    
    # Set global instance for signal handler
    _runner_instance = runner
    
    # Run evaluation
    try:
        runner.run(policy)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt in main! Shutting down...")
        runner.shutdown_requested = True
        runner.stop_event.set()
        # Force shutdown ROS
        if not rospy.is_shutdown():
            rospy.signal_shutdown("KeyboardInterrupt")
    finally:
        logger.info("Evaluation finished. Performing final cleanup...")
        
        # Final memory cleanup
        if hasattr(runner, '_cleanup_memory'):
            runner._cleanup_memory()
        
        # Clear policy from GPU
        if hasattr(policy, 'to'):
            policy.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Delete large objects
        del runner
        if 'policy' in locals():
            del policy
        if 'workspace' in locals():
            del workspace
        
        # Final garbage collection
        for i in range(5):
            collected = gc.collect()
            if collected == 0:
                break
        
        # Log final memory usage
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.info(f"Final memory usage - RSS: {mem_info.rss / 1024 / 1024:.2f} MB, "
                       f"VMS: {mem_info.vms / 1024 / 1024:.2f} MB")
            logger.info("To free swap space, you may need to run: sudo swapoff -a && sudo swapon -a")
        except Exception as e:
            logger.debug(f"Could not get final memory info: {e}")
        
        _runner_instance = None
        logger.info("Cleanup completed. Exiting...")

# def main(cfg):

#     # Get config from task config
#     task_cfg = cfg.task
    
#     # Create custom runner - simplified: single action buffer (no TCP/Gripper separation)
#     # If config has tcp_ensemble_buffer_params, use it; otherwise create default
#     env_runner_cfg = task_cfg.env_runner
#     if 'action_ensemble_buffer_params' in env_runner_cfg:
#         action_ensemble_buffer_params = env_runner_cfg.action_ensemble_buffer_params
#     elif 'tcp_ensemble_buffer_params' in env_runner_cfg:
#         # Fallback: use tcp_ensemble_buffer_params if action_ensemble_buffer_params doesn't exist
#         action_ensemble_buffer_params = env_runner_cfg.tcp_ensemble_buffer_params
#     else:
#         action_ensemble_buffer_params = DictConfig({'ensemble_mode': 'new'})
    
#     if 'latent_action_ensemble_buffer_params' in env_runner_cfg:
#         latent_action_ensemble_buffer_params = env_runner_cfg.latent_action_ensemble_buffer_params
#     elif 'latent_tcp_ensemble_buffer_params' in env_runner_cfg:
#         latent_action_ensemble_buffer_params = env_runner_cfg.latent_tcp_ensemble_buffer_params
#     else:
#         latent_action_ensemble_buffer_params = DictConfig({'ensemble_mode': 'new'})
    
#     # Get action update interval (use tcp_action_update_interval as fallback)
#     action_update_interval = env_runner_cfg.get('action_update_interval', 
#         env_runner_cfg.get('tcp_action_update_interval', 16))
    
#     # Get action clip range (optional)
#     action_clip_range = env_runner_cfg.get('action_clip_range', None)
    
#     # Get joint names from config if available
#     joint_names = env_runner_cfg.get('joint_names', None)
    
#     # Get topic mapping from config if available (maps shape_meta keys to actual ROS topic names)
#     topic_mapping = env_runner_cfg.get('topic_mapping', None)
    
#     if topic_mapping is None:
#         # Default mapping based on common patterns - you can override this in config
#         topic_mapping = {}
#         # Auto-generate mapping for common cases
#         for key in task_cfg.shape_meta.get('obs', {}).keys():
#             if key == 'camera_f':
#                 topic_mapping[key] = '/camera_f/color/image_raw'
#             elif key == 'camera_r':
#                 topic_mapping[key] = '/camera_r/color/image_raw'
#             elif 'joint' in key.lower() and 'state' in key.lower():
#                 # For joint states, try common patterns
#                 if 'right_arm' not in key:
#                     topic_mapping[key] = f'/right_arm/{key}'

#     # Get force subscribe topics from config (topics to subscribe even if not in shape_meta)
#     force_subscribe_topics = env_runner_cfg.get('force_subscribe_topics', None)
#     if force_subscribe_topics is None:
#         # Default: Always subscribe to Marker_Tracking_Right_DXDY
#         force_subscribe_topics = {
#             '/Marker_Tracking_Right_DXDY': 'Float32MultiArray'
#         }
#     else:
#         # Ensure Marker_Tracking_Right_DXDY is always included
#         if '/Marker_Tracking_Right_DXDY' not in force_subscribe_topics:
#             force_subscribe_topics['/Marker_Tracking_Right_DXDY'] = 'Float32MultiArray'
#     shape_meta  = task_cfg.shape_meta
#     shape_meta = dict(shape_meta)
#     obs_subscriber = ROSObservationSubscriber(
#         shape_meta=shape_meta,
#         topic_prefix="/",
#         topic_mapping=topic_mapping,
#         force_subscribe_topics=force_subscribe_topics,
#         max_fps=30,
#         sync_slop=0.4,
#         sync_queue_size=40
#     )
#     output = obs_subscriber.get_obs(2,2)
#     import time

#     logger.info("Waiting for ROS data stream...")
    
#     # 尝试获取数据的最大次数
#     max_retries = 20 
#     output = {}
    
#     for i in range(max_retries):
#         # 尝试获取数据
#         output = obs_subscriber.get_obs(2, 2)
#         # 检查是否成功获取 (字典不为空)
#         if output and len(output) > 0:
#             logger.success(f"Data received after {i+1} attempts!")
#             break
        
#         logger.warning(f"Buffer empty, waiting... ({i+1}/{max_retries})")
#         time.sleep(1.0) # 等待 1 秒让数据飞一会儿

#     if not output:
#         logger.error("Failed to receive any data. Please check if ROS publishers are running!")
#     else:
#         inspect_shapes(output)

# def inspect_shapes(data_dict):
#     print("\n" + "="*30)
#     print(f"{'Key (键名)':<30} | {'Shape (形状)':<20} | {'Dtype (类型)'}")
#     print("-" * 65)
    
#     for key, value in data_dict.items():
#         # 1. 检查是否有 shape 属性 (兼容 numpy, torch, tensorflow)
#         if hasattr(value, 'shape'):
#             shape_str = str(value.shape)
#             dtype_str = str(value.dtype)
#             print(f"{key:<30} | {shape_str:<20} | {dtype_str}")
        
#         # 2. 如果是 List (列表)
#         elif isinstance(value, list):
#             print(f"{key:<30} | Len={len(value):<16} | List")
            
#         # 3. 其他类型 (如 float, int, None)
#         else:
#             print(f"{key:<30} | {'N/A':<20} | {type(value).__name__}")
            
#     print("="*30 + "\n")
#     print(data_dict["Marker_Tracking_Right_DXDY_emb"])


if __name__ == '__main__':
    main()
