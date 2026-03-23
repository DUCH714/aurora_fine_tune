from aurora import AuroraPretrained
from aurora.batch import Batch
import torch
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class AuroraDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return Batch(
            surf_vars=sample["surf_vars"],
            static_vars=sample["static_vars"],
            atmos_vars=sample["atmos_vars"],
        )

def loss_fn(pred, target):
    loss = 0.0
    
    for key in pred.surf_vars:
        loss += ((pred.surf_vars[key] - target.surf_vars[key])**2).mean()
        
    for key in pred.atmos_vars:
        loss += ((pred.atmos_vars[key] - target.atmos_vars[key])**2).mean()
    
    return loss

# 1. 先加载原始模型
base_model = AuroraPretrained(use_lora=False).to(device)
base_model.load_checkpoint()

# 2. 再创建带 LoRA 的模型
model = AuroraPretrained(use_lora=True)

# 3. 加载 backbone 权重（忽略 LoRA）
model.load_state_dict(base_model.state_dict(), strict=False)

model.train()

for name, param in model.named_parameters():
    if "lora" not in name.lower():
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable} / {total}")

# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=1e-4
# )

# for epoch in range(10):
#     for batch in dataloader:
#         batch = batch.to(device)

#         pred = model(batch)

#         loss = loss_fn(pred, batch)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch}, Loss: {loss.item()}")

# torch.save(model.state_dict(), "aurora_lora_finetuned.pt")

# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
