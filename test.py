import torch

print("Torch version:",torch.__version__)
torch.zeros(1).cuda()
print("Is CUDA enabled?",torch.cuda.is_available())