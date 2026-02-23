import torch
from src.model.gpt import TinyGPT

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = TinyGPT()
model = model.to(device)

x = torch.randint(0, 8192, (2, 128)).to(device)
logits = model(x)

print("Logits shape:", logits.shape)
print("Parameter count:", sum(p.numel() for p in model.parameters()))