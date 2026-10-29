"""v2 fish reader: CNN feature grid + one learned query per bit that attends to where its scale is.

v1 had to learn "where is bit k?" inside one dense layer, for every pose and body shape at once.
Here a small FPN-style CNN turns the 400x640 plate into a 50x80 grid of 128-d features (1 token
~ 1 scale), and a 3-layer transformer decoder runs 248 bit queries over it: self-attention lets
the queries agree on the fish's pose, cross-attention lets each one look up its own scale. The
last cross-attention map of each query is literally where that bit looks.

python train2.py [train_glob val.npz out.pt steps]   (defaults: /content/...)
"""
import sys, os, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '/content')  # kernel exec
from train import augment, load, block, dev  # noqa: E402


def pos2d(h, w, d, device):
    """Fixed 2-D sine/cosine positional encoding -> (h*w, d)."""
    q = d // 4
    f = 1 / (100 ** (torch.arange(q, device=device) / q))
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing='ij')
    py, px = (y.flatten()[:, None] * f * 2 * math.pi), (x.flatten()[:, None] * f * 2 * math.pi)
    return torch.cat([py.sin(), py.cos(), px.sin(), px.cos()], 1)


class Backbone(nn.Module):
    """Plate -> stride-8 feature grid, with stride-16/32 context folded back in (FPN top-down)."""

    def __init__(self, d=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(block(32, 48, 1), block(48, 64, 2))   # /4
        self.c3, self.c4, self.c5 = block(64, 128, 2), block(128, 192, 2), block(192, 256, 2)  # /8 /16 /32
        self.l3, self.l4, self.l5 = nn.Conv2d(128, d, 1), nn.Conv2d(192, d, 1), nn.Conv2d(256, d, 1)
        self.fuse = nn.Sequential(nn.Conv2d(d, d, 3, 1, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True))

    def forward(self, x):
        c3 = self.c3(self.c2(self.stem(x)))
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[2:], mode='bilinear', align_corners=False)
        p4 = self.l4(c4) + up(self.l5(c5), c4)
        return self.fuse(self.l3(c3) + up(p4, c3))


class DecLayer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, h, batch_first=True)
        self.ca = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, qpos, mem, mpos, keep=False):
        t = self.n1(q) + qpos
        q = q + self.sa(t, t, self.n1(q), need_weights=False)[0]
        a, w = self.ca(self.n2(q) + qpos, mem + mpos, mem, need_weights=keep)
        q = q + a
        return q + self.ff(self.n3(q)), w


class BitReader(nn.Module):
    def __init__(self, nbits, d=128, heads=4, layers=3):
        super().__init__()
        self.backbone = Backbone(d)
        self.qpos = nn.Parameter(torch.randn(nbits, d) * .02)  # one learned query per bit
        self.dec = nn.ModuleList([DecLayer(d, heads) for _ in range(layers)])
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, 1)

    def forward(self, x, attn=False):
        """x: (B,1,H,W) in [0,1] -> logits (B, nbits) [, last cross-attention (B, nbits, h, w)]."""
        B = x.shape[0]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, x.shape[2], device=x.device),
                                torch.linspace(-1, 1, x.shape[3], device=x.device), indexing='ij')
        f = self.backbone(torch.cat([1 - x, torch.stack([xs, ys])[None].expand(B, 2, -1, -1)], 1))
        h, w = f.shape[2:]
        mem = f.flatten(2).transpose(1, 2)
        mpos = pos2d(h, w, mem.shape[2], x.device)[None]
        qpos = self.qpos[None].expand(B, -1, -1)
        q = torch.zeros_like(qpos)
        for i, layer in enumerate(self.dec):
            q, wts = layer(q, qpos, mem, mpos, keep=attn and i == len(self.dec) - 1)
        logits = self.out(self.norm(q)).squeeze(-1)
        return (logits, wts.view(B, -1, h, w)) if attn else logits


if __name__ == '__main__':
    A = sys.argv[1:] if len(sys.argv) > 4 else ['/content/train2_*.npz', '/content/val2.npz', '/content/fishnn2.pt', '40000']
    TRAIN, VAL, OUT, STEPS = A[0], A[1], A[2], int(A[3])
    Xtr, Ytr = load(TRAIN)
    Xva, Yva = load(VAL)
    nbits = Ytr.shape[1]
    print(f'train {tuple(Xtr.shape)} val {tuple(Xva.shape)} bits {nbits} device {dev} '
          f'{torch.cuda.get_device_name(0) if dev == "cuda" else ""}', flush=True)
    net = BitReader(nbits).to(dev)
    print(f'params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=STEPS, pct_start=.05)
    start = 1
    if os.path.exists(OUT):  # resume after a lost VM
        ck = torch.load(OUT, map_location=dev)
        net.load_state_dict(ck['state']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start = ck['step'] + 1
        print(f'resumed at step {start}', flush=True)
    BS, t0 = 64, time.time()
    amp = dict(device_type='cuda', dtype=torch.bfloat16) if dev == 'cuda' else dict(device_type='cpu', enabled=False)

    def evaluate(s):
        net.eval()
        g = torch.Generator(device=dev).manual_seed(0)
        acc = []
        with torch.no_grad(), torch.autocast(**amp):
            for i in range(0, len(Xva), 64):
                pred = (net(augment(Xva[i:i + 64, None].float() / 255, s, g)) > 0).float()
                acc.append((pred == Yva[i:i + 64]).float().mean(1))
        net.train()
        acc = torch.cat(acc)
        return acc.mean().item(), (acc > .985).float().mean().item()  # ~<=3 bit errors: inside RS reach

    for step in range(start, STEPS + 1):
        s = min(1., step / (.3 * STEPS))
        idx = torch.randint(0, len(Xtr), (BS,), device=dev)
        x = augment(Xtr[idx, None].float() / 255, s)
        with torch.autocast(**amp):
            loss = F.binary_cross_entropy_with_logits(net(x).float(), Ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 200:
            (a0, e0), (a1, e1), (a5, e5) = evaluate(0.), evaluate(1.), evaluate(.5)
            print(f'step {step:6d}  s {s:.2f}  loss {loss.item():.4f}  val acc clean {a0:.4f} (ok {e0:.2f}) | '
                  f'mid-aug {a5:.4f} (ok {e5:.2f}) | full-aug {a1:.4f} (ok {e1:.2f})  '
                  f'{(time.time() - t0) / (step - start + 1):.3f}s/step', flush=True)
        if step % 2000 == 0 or step == STEPS:
            torch.save({'state': net.state_dict(), 'nbits': nbits, 'step': step,
                        'opt': opt.state_dict(), 'sched': sched.state_dict()}, OUT)
    print('done', flush=True)
