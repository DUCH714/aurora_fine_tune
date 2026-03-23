from aurora import AuroraPretrained
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AuroraPretrained().to(device)
model.load_checkpoint()

model.eval()
