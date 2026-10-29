"""Train a CNN to read fishdraw plates end to end: 320x512 grayscale -> CODE_BITS bits.

Meant for a Colab GPU. Every step augments on the GPU (homography, blur, resolution loss,
lighting, noise, JPEG-like block quantisation), with strength ramped up over the first 30%
of training so the net first learns to read, then to read through damage.

python train.py [train.npz val.npz out.pt steps]   (defaults: /content/...)
"""
import sys, os, time, math, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(pattern):
    """One .npz or a glob of shards -> uint8 images (N, H, W) and float bit labels, on the device."""
    ds = [np.load(p) for p in sorted(glob.glob(pattern))]
    X = torch.from_numpy(np.concatenate([d['X'] for d in ds])).to(dev)
    Y = np.concatenate([np.unpackbits(d['Y'], axis=1)[:, :int(d['code_bits'])] for d in ds])
    return X, torch.from_numpy(Y).float().to(dev)


# --- augmentation ------------------------------------------------------------------------
_DCT = torch.tensor([[math.sqrt((1 if k == 0 else 2) / 8) * math.cos((2 * n + 1) * k * math.pi / 16)
                      for n in range(8)] for k in range(8)], device=dev)
_FREQ = 1 + (torch.arange(8, device=dev)[:, None] + torch.arange(8, device=dev)[None]).float() / 2


def augment(x, s, g=None):
    """x: (B,1,H,W) in [0,1], H and W multiples of 8; s: strength 0..1."""
    B, _, H, W = x.shape
    r = lambda *sh: torch.rand(*sh, device=dev, generator=g)
    # homography from output to input: random rotation/zoom/shift + corner jitter
    a = W / H
    src = torch.tensor([[-a, -1], [a, -1], [a, 1], [-a, 1]], device=dev).expand(B, 4, 2)
    th = (r(B) * 2 - 1) * math.radians(30) * s
    k = torch.exp((r(B) * .55 - .05) * s)                     # zoom out up to ~1.7x (content shrinks)
    R = torch.stack([torch.cos(th), -torch.sin(th), torch.sin(th), torch.cos(th)], -1).view(B, 2, 2)
    dst = k[:, None, None] * src @ R.transpose(1, 2) + (r(B, 1, 2) * 2 - 1) * .12 * s + (r(B, 4, 2) * 2 - 1) * .1 * s
    src_n, dst_n = src / torch.tensor([a, 1.], device=dev), dst / torch.tensor([a, 1.], device=dev)
    rows = []
    for i in range(4):
        (u, v), (X_, Y_) = src_n[:, i].T, dst_n[:, i].T
        z, o = torch.zeros_like(u), torch.ones_like(u)
        rows += [torch.stack([u, v, o, z, z, z, -u * X_, -v * X_], -1), torch.stack([z, z, z, u, v, o, -u * Y_, -v * Y_], -1)]
    rhs = torch.stack([dst_n[:, i, j] for i in range(4) for j in range(2)], -1)
    h = torch.linalg.solve(torch.stack(rows, 1), rhs)
    Hm = torch.cat([h, torch.ones(B, 1, device=dev)], -1).view(B, 3, 3)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing='ij')
    p = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(1, -1, 3) @ Hm.transpose(1, 2)
    grid = (p[..., :2] / p[..., 2:]).view(B, H, W, 2)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
    # resolution loss
    f = 1 - .6 * s * float(r(1))
    if f < .97:
        x = F.interpolate(F.interpolate(x, scale_factor=f, mode='area'), size=(H, W), mode='bilinear', align_corners=False)
    # blur (per-sample sigma)
    sig = (r(B) * 1.8 * s).clamp(min=1e-3)
    t = torch.arange(-4, 5, device=dev).float()
    ker = torch.exp(-t[None] ** 2 / (2 * sig[:, None] ** 2))
    ker = (ker / ker.sum(1, keepdim=True)).view(B, 1, 1, 9)
    x = F.conv2d(x.view(1, B, H, W), ker, padding=(0, 4), groups=B)
    x = F.conv2d(x, ker.transpose(2, 3), padding=(4, 0), groups=B).view(B, 1, H, W)
    # lighting: shadow ramp, contrast, offset
    ang = r(B, 1, 1) * 2 * math.pi
    ramp = (xs[None] * torch.cos(ang) + ys[None] * torch.sin(ang)).clamp(-1, 1)
    x = x * (1 - r(B, 1, 1) * .6 * s * (.5 + .5 * ramp))[:, None]
    x = x * (1 - r(B, 1, 1, 1) * .35 * s) + (r(B, 1, 1, 1) * 2 - 1) * .05 * s
    # sensor noise
    x = x + torch.randn(x.shape, device=dev, generator=g) * r(B, 1, 1, 1) * .07 * s
    # JPEG-like: quantise 8x8 block DCT coefficients, coarser at high frequencies
    if float(r(1)) < .6:
        blk = x.view(B, 1, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 3, 5)
        c = _DCT @ blk @ _DCT.T
        q = (.004 + r(B, 1, 1, 1, 1, 1) * .06 * s) * _FREQ
        c = torch.round(c / q) * q
        x = (_DCT.T @ c @ _DCT).permute(0, 1, 2, 4, 3, 5).reshape(B, 1, H, W)
    return x.clamp(0, 1)


# --- model -------------------------------------------------------------------------------
def block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Net(nn.Module):
    """Plain conv trunk to 10x16, then a dense head: the head is where each bit learns *where* to look."""

    def __init__(self, nbits):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(block(32, 48, 1), block(48, 64, 2), block(64, 96, 2), block(96, 128, 2), block(128, 192, 2))
        self.head = nn.Sequential(nn.Conv2d(192, 48, 1), nn.ReLU(inplace=True), nn.Flatten(),
                                  nn.Linear(48 * 10 * 16, 1024), nn.ReLU(inplace=True), nn.Dropout(.1), nn.Linear(1024, nbits))

    def forward(self, x):  # x: (B,1,H,W) in [0,1]
        B = x.shape[0]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, x.shape[2], device=x.device),
                                torch.linspace(-1, 1, x.shape[3], device=x.device), indexing='ij')
        coords = torch.stack([xs, ys])[None].expand(B, 2, -1, -1)  # CoordConv: lets filters know where they are
        return self.head(self.trunk(self.stem(torch.cat([1 - x, coords], 1))))


if __name__ == '__main__':
    A = sys.argv[1:] if len(sys.argv) > 4 else ['/content/train_*.npz', '/content/val.npz', '/content/fishnn.pt', '30000']
    TRAIN, VAL, OUT, STEPS = A[0], A[1], A[2], int(A[3])
    Xtr, Ytr = load(TRAIN)
    Xva, Yva = load(VAL)
    nbits = Ytr.shape[1]
    print(f'train {tuple(Xtr.shape)} val {tuple(Xva.shape)} bits {nbits} device {dev}', flush=True)
    net = Net(nbits).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=.03)
    scaler = torch.amp.GradScaler()
    start = 1
    if os.path.exists(OUT):  # resume after a lost VM: weights, optimiser, schedule and step
        ck = torch.load(OUT, map_location=dev)
        if 'step' in ck:
            net.load_state_dict(ck['state']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
            start = ck['step'] + 1
            print(f'resumed at step {start}', flush=True)
    BS, t0 = 32, time.time()

    def evaluate(s):
        net.eval()
        g = torch.Generator(device=dev).manual_seed(0)
        acc, exact = [], []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16):
            for i in range(0, len(Xva), 64):
                x = augment(Xva[i:i + 64, None].float() / 255, s, g)
                pred = (net(x) > 0).float()
                acc.append((pred == Yva[i:i + 64]).float().mean(1))
        net.train()
        acc = torch.cat(acc)
        return acc.mean().item(), (acc > .985).float().mean().item()  # ~<=3 bit errors: inside RS reach

    for step in range(start, STEPS + 1):
        s = min(1., step / (.3 * STEPS))
        idx = torch.randint(0, len(Xtr), (BS,), device=dev)
        x = augment(Xtr[idx, None].float() / 255, s)
        with torch.autocast(dev, dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(net(x.contiguous(memory_format=torch.channels_last)), Ytr[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 1000 == 0 or step == 200:
            (a0, e0), (a1, e1) = evaluate(0.), evaluate(1.)
            print(f'step {step:6d}  s {s:.2f}  loss {loss.item():.4f}  val clean acc {a0:.4f} ok {e0:.2f} | '
                  f'full-aug acc {a1:.4f} ok {e1:.2f}  {(time.time() - t0) / (step - start + 1):.3f}s/step', flush=True)
        if step % 2000 == 0 or step == STEPS:
            torch.save({'state': net.state_dict(), 'nbits': nbits, 'step': step,
                        'opt': opt.state_dict(), 'sched': sched.state_dict()}, OUT)
    print('done', flush=True)
