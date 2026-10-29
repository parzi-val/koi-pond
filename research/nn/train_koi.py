"""Koi strip reader: v2's BitReader (FPN grid + one learned query per bit) on unbent koi strips.

The strips (koi_data.py) are already in body coordinates, so the net doesn't have to find the koi,
only to read it through whatever the photo did: residual misalignment from the spine fit,
blur, noise, JPEG. 128x640 strips -> 16x80 token grid -> 176 bit queries.

Curriculum: the first EASY fraction of steps sees only mildly damaged views as photographed; after
that, every view plus extra augmentation in CPU dataloader workers (fixed shapes, real JPEG, nothing
data-dependent on the accelerator, so a compiled TPU graph is reused every step): small affine jitter
standing in for spine-fit error, then blur/gain/noise/JPEG. Thrown in at full difficulty from step one,
the net settled on the constant bits (koi headers) and never found the hatches.
Runs on a TPU (Colab's torch_tpu), a GPU or the CPU.

python train_koi.py [train_glob val_glob out.pt steps]   (defaults: /content/...)
"""
import sys, os, time, glob
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils._pytree import tree_map

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '/content/research/nn'
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from train2 import BitReader  # noqa: E402
import koi  # noqa: E402
from reedsolo import ReedSolomonError  # noqa: E402

try:
    import torch_tpu  # noqa: F401  (Colab TPU runtimes: registers the 'tpu' device)
    dev = torch.device('tpu')
except ImportError:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BS = int(os.environ.get('KOI_BS', 128))  # small for smoke tests: drop_last on a tiny set gives no batches at all
EASY, EASY_SEV = .15, .35  # curriculum: share of steps on easy views, and what counts as easy


def load(pattern):
    """Shards -> ink strips (uint8 N x H x W, ink bright), bit labels (uint8 N x 176), view severity, head-end
    picked wrong (rev), classical bits or None."""
    ds = [np.load(p) for p in sorted(glob.glob(pattern))]
    X = np.concatenate([d['X'] for d in ds])
    Y = np.unpackbits(np.concatenate([d['Y'] for d in ds]), axis=1)
    sev, rev = np.concatenate([d['sev'] for d in ds]), np.concatenate([d['rev'] for d in ds])
    cls = [d['cls'] for d in ds]
    cls = np.unpackbits(np.concatenate(cls), axis=1) if all(len(c) == len(d['X']) for c, d in zip(cls, ds)) else None
    return X, Y, sev, rev, cls


def jitter(x, rng):
    """One ink strip -> a slightly worse photo of it (strength drawn per strip)."""
    h, w = x.shape
    s = rng.uniform()
    su, sv = 1 + rng.uniform(-.03, .03) * s, 1 + rng.uniform(-.08, .08) * s  # length / half-width misjudged
    sh = np.tan(np.radians(rng.uniform(-1.5, 1.5) * s)) * w / h             # spine tilted against the body
    M = np.float32([[su, 0, (1 - su) * w / 2 + rng.uniform(-8, 8) * s],
                    [sh * h / w, sv, (1 - sv) * h / 2 - sh * h / 2 + rng.uniform(-6, 6) * s]])
    x = cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.uniform() < .7:
        x = cv2.GaussianBlur(x, (0, 0), .2 + 1.2 * s * rng.uniform())
    x = x * rng.uniform(1 - .4 * s, 1) + rng.normal(0, 12 * s * rng.uniform(), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.uniform() < .5:
        x = cv2.imdecode(cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, int(95 - 60 * s * rng.uniform())])[1], 0)
    return x


class Strips(torch.utils.data.Dataset):
    """uint8 ink strips (4x less to copy to the device than floats); to_input() finishes on the device."""
    def __init__(self, X, Y, aug, idx=None):
        self.X, self.Y, self.aug = X, Y, aug
        self.idx = np.arange(len(X)) if idx is None else idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = self.X[self.idx[i]]
        if self.aug:
            x = jitter(x, np.random.default_rng())
        return torch.from_numpy(x)[None], torch.from_numpy(self.Y[self.idx[i]])


def to_input(x):
    return 1 - x.float() / 255  # ink map -> paper-white image, as BitReader expects


def inner_ok(bits, Y):
    """Fraction of strips whose 176 bits pass the koi's inner RS(4) and give back the true chunk."""
    ok = 0
    for b, y in zip(bits.astype(np.uint8), Y.astype(np.uint8)):
        try:
            ok += bytes(koi._RS.decode(np.packbits(b).tobytes())[0]) == np.packbits(y)[:-4].tobytes()
        except ReedSolomonError:
            pass
    return ok / len(bits)


def _worker_init(_):
    cv2.setNumThreads(1)


def endless(ds, workers):
    """Shuffled batches forever. Persistent workers, forked once by the first next(): forking again mid-run,
    with the TPU runtime's threads busy, left the new workers deadlocked and training silently stalled."""
    dl = torch.utils.data.DataLoader(ds, BS, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=workers > 0,
                                     prefetch_factor=4 if workers else None, worker_init_fn=_worker_init)
    while True:
        yield from dl


if __name__ == '__main__':
    A = sys.argv[1:] if len(sys.argv) > 4 else ['/content/data/koi_train_*.npz', '/content/data/koi_val_*.npz',
                                                '/content/koinn.pt', '40000']
    TRAIN, VAL, OUT, STEPS = A[0], A[1], A[2], int(A[3])
    Xtr, Ytr, Str, Rtr, _ = load(TRAIN)
    Xva, Yva, _, _, Cva = load(VAL)
    nbits = Ytr.shape[1]
    workers = min(32, (os.cpu_count() or 2) - 2)
    easy = np.nonzero(Str < EASY_SEV)[0]
    print(f'train {Xtr.shape} ({len(easy)} easy, head end picked wrong on {Rtr.mean():.2%}) val {Xva.shape} '
          f'bits {nbits} device {dev} workers {workers}', flush=True)
    if Cva is not None:
        print(f'classical on val: bit acc {(Cva == Yva).mean():.4f}  inner RS ok {inner_ok(Cva, Yva):.3f}', flush=True)
    net = BitReader(nbits).to(dev)
    print(f'params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=STEPS, pct_start=.05)
    step = 0
    if os.path.exists(OUT):  # resume after a lost VM
        ck = torch.load(OUT, map_location='cpu')
        net.load_state_dict(ck['state']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        step = ck['step']
        print(f'resumed at step {step + 1}', flush=True)
    run = torch.compile(net, backend='tpu') if dev.type == 'tpu' else net  # TPU eager is ~20x slower than compiled
    va = torch.utils.data.DataLoader(Strips(Xva, Yva, False), BS)  # in-process: no forking during training
    easy_b = endless(Strips(Xtr, Ytr, False, easy), workers) if len(easy) >= BS else None
    full_b = endless(Strips(Xtr, Ytr, True), workers)
    first = [next(b) for b in (easy_b, full_b) if b is not None]  # fork every worker now, before the TPU gets busy

    def batches():
        yield from first[:1] if easy_b is not None and step < EASY * STEPS else first[-1:]
        while True:
            yield next(easy_b) if easy_b is not None and step < EASY * STEPS else next(full_b)

    def evaluate():
        """Val strips exactly as photographed (no extra augmentation): bit accuracy and inner-RS success.
        Stays in train mode (batch statistics): compiled on the TPU, BatchNorm's running stats are never
        written back, so eval mode would be meaningless. koi_eval.py fixes the stats for deployment."""
        pred = []
        with torch.no_grad():
            for x, _ in va:
                pred.append((run(to_input(x.to(dev))) > 0).cpu())
        pred = torch.cat(pred).numpy()
        return (pred == Yva).mean(), inner_ok(pred, Yva)

    def save():
        ck = {'state': net.state_dict(), 'nbits': nbits, 'step': step, 'opt': opt.state_dict(), 'sched': sched.state_dict()}
        torch.save(tree_map(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, ck), OUT)

    t0, done = time.time(), step
    for x, y in batches():
        step += 1
        x, y = to_input(x.to(dev, non_blocking=True)), y.to(dev, non_blocking=True).float()
        loss = F.binary_cross_entropy_with_logits(run(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        sched.step()
        if step % 10 == 0 and dev.type != 'cpu':  # TPU work is queued lazily: an unbounded queue holds every batch's buffers
            torch.accelerator.synchronize()
        if step % 1000 == 0 or step == 200:
            acc, ok = evaluate()
            print(f'step {step:6d}  loss {loss.item():.4f}  val bit acc {acc:.4f}  inner RS ok {ok:.3f}  '
                  f'{(time.time() - t0) / (step - done):.3f}s/step', flush=True)
        if step % 2000 == 0 or step == STEPS:
            save()
        if step == STEPS:
            break
    print('done', flush=True)
