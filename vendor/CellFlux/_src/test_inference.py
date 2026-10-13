"""Minimal CellFlux inference smoke test: pure-noise ODE sampling."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.unet import UNetModel
from models.model_configs import MODEL_CONFIGS

CKPT = Path('D:/CellPainting-Claw-main2/CellFlux/checkpoint.pth')
EMB = Path('D:/CellPainting-Claw-main2/CellFlux/emb_fp.csv')


class CFGScaledModel(ModelWrapper):
    """Classifier-free-guidance wrapper mirroring CellFlux's eval_loop."""

    def forward(self, x, t, cfg_scale, extra):
        t = torch.zeros(x.shape[0], device=x.device) + t
        if cfg_scale != 0.0:
            conditional = self.model(x, t, extra=extra)
            condition_free = self.model(x, t, extra={})
            return (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        return self.model(x, t, extra=extra)


def load_fp(path):
    table = {}
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row and row[0]:
                table[row[0]] = [float(v) for v in row[1:]]
    return table


def main():
    # 1. model
    model = UNetModel(**MODEL_CONFIGS['bbbc021'])
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    # EMA wrapper state dict uses a "model." prefix plus a "num_updates" counter.
    state_dict = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state_dict)
    model.eval()

    # 2. fingerprint embedding (1024-dim)
    table = load_fp(EMB)
    print('compounds:', len(table), '| embed dim:', len(next(iter(table.values()))))
    z = torch.tensor(table['taxol'], dtype=torch.float32).unsqueeze(0)  # (1, 1024)

    # 3. pure-noise start (use_initial=0)
    torch.manual_seed(42)
    x_0 = torch.randn(1, 3, 128, 128)

    # 4. ODE sampling (heun2, 50 steps)
    cfg_model = CFGScaledModel(model)
    solver = ODESolver(velocity_model=cfg_model)
    time_grid = torch.linspace(0, 1, 50)
    samples = solver.sample(
        x_init=x_0,
        step_size=None,
        method='heun2',
        time_grid=time_grid,
        cfg_scale=0.2,
        extra={'concat_conditioning': z},
    )
    print('samples shape:', samples.shape)

    import numpy as np
    from PIL import Image
    img_np = (torch.clamp(samples[0] * 0.5 + 0.5, 0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
    out = Path('D:/CellPainting-Claw-main2/CellFlux/test_taxol.png')
    Image.fromarray(img_np, 'RGB').save(out)
    print('saved:', out)


if __name__ == '__main__':
    main()
