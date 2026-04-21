import pickle
import numpy as np
import torch
import sys
from types import SimpleNamespace # 引入 SimpleNamespace 支持

def print_structure(data, indent=0):
    prefix = "  " * indent
    
    # 1. 处理字典 (Dict)
    if isinstance(data, dict):
        print(f"{prefix}Type: dict, Keys: {len(data.keys())}")
        for key, value in data.items():
            print(f"{prefix}├── Key: '{key}'")
            print_structure(value, indent + 1)

    # 2. [新增] 处理 SimpleNamespace (你的情况)
    elif isinstance(data, SimpleNamespace):
        # 将其转换为字典来查看
        attrs = vars(data) 
        print(f"{prefix}Type: SimpleNamespace, Attributes: {len(attrs)}")
        for key, value in attrs.items():
            print(f"{prefix}├── Attribute: '{key}'")
            print_structure(value, indent + 1)
            
    # 3. 处理列表/元组
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}Type: {type(data).__name__}, Length: {len(data)}")
        if len(data) > 0:
            print(f"{prefix}├── [Element 0 Sample]:")
            print_structure(data[0], indent + 1)
            
    # 4. 处理 Numpy
    elif isinstance(data, np.ndarray):
        print(f"{prefix}Type: numpy.ndarray, Shape: {data.shape}, Dtype: {data.dtype}")
        
    # 5. 处理 Tensor
    elif torch.is_tensor(data):
        print(f"{prefix}Type: torch.Tensor, Shape: {tuple(data.shape)}, Dtype: {data.dtype}")
        
    else:
        print(f"{prefix}Type: {type(data).__name__}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "output.pkl"
    print(f"--- Inspecting: {file_path} ---")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print_structure(data)
    except Exception as e:
        print(f"Error: {e}")