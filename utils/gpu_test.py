import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

devNumber = torch.cuda.current_device()

devName = torch.cuda.get_device_name(devNumber)

print(f"Using device: {device}")

print(f"Device number: {devNumber}")

print(f"Device name: {devName}")