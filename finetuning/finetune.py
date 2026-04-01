"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch

from aurora import AuroraPretrained, Batch, Metadata

import xarray as xr
from huggingface_hub import hf_hub_download
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

def loss(pred: Batch) -> torch.Tensor:
    """A sample loss function. You should replace this with your own loss function."""
    surf_values = pred.surf_vars.values()
    atmos_values = pred.atmos_vars.values()
    return sum((x * x).sum() for x in tuple(surf_values) + tuple(atmos_values))


base_model = AuroraPretrained(autocast=True)
base_model.load_checkpoint()
model = AuroraPretrained(use_lora=True, autocast=True)
model.load_state_dict(base_model.state_dict(), strict=False)
model.train()
model.configure_activation_checkpointing()
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

zarr_path = '/mnt/era5/era5.zarr/'
xarr = xr.open_dataset(zarr_path, engine="zarr", chunks="auto")
xarr = xarr.sel(time=slice('2019-01-01', '2019-01-10'))

surf_keys = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",  # 33.9% NaN
    "sp": "surface_pressure",
    "tp6h": "total precipitation_6hr",
    "tcc": "total_cloud_cover",
    "sd": "snow_depth",
    "sst": "sea_surface_temperature",
    "siconc": "sea_ice_cover"
}

atmos_keys = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential"
}

static_keys = {
    "z": "geopotential_at_surface",  # 33.9% NaN
    "lsm": "land_sea_mask",          # 33.9% NaN
    "slt": "soil_type"
}


def _time_window_to_tensor(var_name: str, t0: int, t1: int) -> torch.Tensor:
    # Aurora expects surf vars as (B, H, Lat, Lon) and atmos vars as (B, H, L, Lat, Lon).
    return torch.tensor(xarr[var_name].isel(time=slice(t0, t1 + 1)).values[None], dtype=torch.float32)


def _get_levels() -> tuple[int, ...]:
    if "level" in xarr.coords:
        return tuple(int(v) for v in xarr.level.values.tolist())
    if "pressure_level" in xarr.coords:
        return tuple(int(v) for v in xarr.pressure_level.values.tolist())
    raise KeyError("Could not find level coordinate. Expected 'level' or 'pressure_level'.")


def _build_static_vars() -> dict[str, torch.Tensor]:
    lat_n = xarr.sizes["latitude"]
    lon_n = xarr.sizes["longitude"]

    # Some ERA5 exports do not include Aurora static vars; fall back to zeros to keep the loop runnable.
    static_vars: dict[str, torch.Tensor] = {
        "lsm": torch.zeros(lat_n, lon_n, dtype=torch.float32),
        "z": torch.zeros(lat_n, lon_n, dtype=torch.float32),
        "slt": torch.zeros(lat_n, lon_n, dtype=torch.float32),
    }

    for aurora_name, ds_name in static_keys.items():
        if ds_name in xarr.data_vars:
            data = xarr[ds_name]
            if "time" in data.dims:
                static_vars[aurora_name] = torch.tensor(data.isel(time=0).values, dtype=torch.float32)
            else:
                static_vars[aurora_name] = torch.tensor(data.values, dtype=torch.float32)

    return static_vars


levels = _get_levels()
static_path = hf_hub_download(repo_id="microsoft/aurora",filename="aurora-0.25-wave-static.pickle",)
with open(static_path, "rb") as f:
    static_vars = pickle.load(f)
# static_vars = _build_static_vars()
num_time_steps = xarr.sizes["time"]
max_steps = min(10, num_time_steps - 1)

print('starting training loop')
for i in range(max_steps):
    print(f"Step {i}")

    t0, t1 = i, i + 1
    batch = Batch(
        surf_vars={
            k: _time_window_to_tensor(v, t0=t0, t1=t1)
            for k, v in surf_keys.items()
            if k in ("2t", "10u", "10v", "msl")
        },
        static_vars=static_vars,
        atmos_vars={
            k: _time_window_to_tensor(v, t0=t0, t1=t1)
            for k, v in atmos_keys.items()
            if k in ("z", "u", "v", "t", "q")
        },
        metadata=Metadata(
            lat=torch.tensor(xarr.latitude.values),
            lon=torch.tensor(xarr.longitude.values),
            time=(xarr.time.values[t1].astype("datetime64[s]").tolist(),),
            atmos_levels=levels,
        ),
    )

    opt.zero_grad()
    prediction = model(batch.to(device))
    loss_value = loss(prediction)
    loss_value.backward()
    opt.step()

    print(f"Step {i}, Loss: {loss_value.item():.3e}")

torch.save(model.state_dict(), "aurora_lora_finetuned.pt")