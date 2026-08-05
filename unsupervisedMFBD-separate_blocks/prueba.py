import torch
from pathlib import Path

ckpt_path = Path("/scratch/paulabp/TFM/run_outputs_v5_50_acc_sched/best_model.pt")
print(f"Existe checkpoint: {ckpt_path.exists()}")

checkpoint = torch.load(ckpt_path, map_location='cpu')
print("Claves disponibles en el checkpoint:", checkpoint.keys() if isinstance(checkpoint, dict) else "Es un state_dict directo")