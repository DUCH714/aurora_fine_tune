"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""
from datetime import datetime

import torch

from aurora import AuroraPretrained, Batch, Metadata

device = "cuda:3" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

def loss(pred: Batch) -> torch.Tensor:
    """A sample loss function. You should replace this with your own loss function."""
    surf_values = prediction.surf_vars.values()
    atmos_values = prediction.atmos_vars.values()
    return sum((x * x).sum() for x in tuple(surf_values) + tuple(atmos_values))


base_model = AuroraPretrained(autocast=True)
base_model.load_checkpoint()
model = AuroraPretrained(use_lora=True)
model.load_state_dict(base_model.state_dict(), strict=False)
model.train()
model = model.to(device)

for name, param in model.named_parameters():
    if "lora" not in name.lower():
        param.requires_grad = False

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable rate: {trainable} / {total}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(10):
    print(f"Step {i}")

    # Train on random data. You should replace this with your own data.
    batch = Batch(
        surf_vars={k: torch.randn(1, 2, 721, 1440) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(721, 1440) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 13, 721, 1440) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 721),
            lon=torch.linspace(0, 360, 1440 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        ),
    )

    opt.zero_grad()
    prediction = model(batch.to(device))
    loss_value = loss(prediction)
    loss_value.backward()
    opt.step()

    print(f"Step {i}, Loss: {loss_value.item():.3e}")

torch.save(model.state_dict(), "aurora_lora_finetuned.pt")