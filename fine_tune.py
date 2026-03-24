from aurora import AuroraPretrained
from aurora.batch import Batch
import torch
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class AuroraDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

        # 可以预定义（推荐）
        self.metadata = {
            "lat": torch.linspace(-90, 90, 721),
            "lon": torch.linspace(0, 360, 1440),
            "time": torch.tensor([0]),
            "lead_time": torch.tensor([0]),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return Batch(
            surf_vars=sample["surf_vars"],
            static_vars=sample["static_vars"],
            atmos_vars=sample["atmos_vars"],
            metadata=self.metadata   # ✅ 加上这个
        )

def collate_fn(batch_list):
    surf_vars = {}
    atmos_vars = {}
    static_vars = {}

    # 拼接 surf_vars
    for key in batch_list[0].surf_vars:
        surf_vars[key] = torch.stack([
            b.surf_vars[key] for b in batch_list
        ])

    # 拼接 atmos_vars
    for key in batch_list[0].atmos_vars:
        atmos_vars[key] = torch.stack([
            b.atmos_vars[key] for b in batch_list
        ])

    # static_vars（通常不随batch变）
    for key in batch_list[0].static_vars:
        static_vars[key] = torch.stack([
            b.static_vars[key] for b in batch_list
        ])

    # metadata（⚠️ 一般直接取第一个）
    metadata = batch_list[0].metadata

    return Batch(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
        metadata=metadata
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

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable rate: {trainable} / {total}")

data_list = [
    {
        "surf_vars": {
            "2t": torch.randn(1, 721, 1440),
        },
        "static_vars": {
            "lsm": torch.randn(1, 721, 1440),
        },
        "atmos_vars": {
            "t": torch.randn(13, 721, 1440),
        },
    }
]

dataset = AuroraDataset(data_list)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True,collate_fn=collate_fn)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

for epoch in range(10):
    for batch in dataloader:
        batch = batch.to(device)

        pred = model(batch)

        loss = loss_fn(pred, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "aurora_lora_finetuned.pt")

