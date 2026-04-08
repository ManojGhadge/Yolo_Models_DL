# system_info.py
import torch
import platform

print("=" * 40)
print("SYSTEM CONFIGURATION")
print("=" * 40)
print(f"OS            : {platform.system()} {platform.release()}")
print(f"Python        : {platform.python_version()}")
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device        : {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 40)