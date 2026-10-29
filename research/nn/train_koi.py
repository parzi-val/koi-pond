"""Koi strip reader: v2's BitReader (FPN grid + one learned query per bit) on unbent koi strips.

The strips (koi_data.py) are already in body coordinates, so the net doesn't have to find the koi,
only to read it through whatever the photo did: residual misalignment from the spine fit,
blur, noise, JPEG. 128x640 strips -> 16x80 token grid -> 176 bit queries.

Training adds mild GPU augmentation on top of the photographed views (small homography =
spine-fit error, plus blur/noise/JPEG), and turns half the strips 180 degrees with the label
unchanged, so the reader doesn't depend on the classical head-end guess.

python train_koi.py [train_glob val_glob out.pt steps]   (defaults: /content/...)
"""
import sys, os, time, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '/content'
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from train import augment, dev  # noqa: E402
from train2 import BitReader  # noqa: E402
import koi  # noqa: E402
from reedsolo import ReedSolomonError  # noqa: E402

AUG = .35  # augment() strength cap: +-10 deg, small zoom/shift, light blur/noise/JPEG


def load(pattern):
    """Shards -> strips as paper-white floats source (uint8, N x H x W), bit labels, classical bits (or None)."""
    ds = [np.load(p) for p in sorted(glob.glob(pattern))]
    X = torch.from_numpy(np.concatenate([d['X'] for d in ds])).to(dev)
    Y = torch.from_numpy(np.unpackbits(np.concatenate([d['Y'] for d in ds]), axis=1)).float().to(dev)
    cls = [d['cls'] for d in ds]
    cls = np.unpackbits(np.concatenate(cls), axis=1) if all(len(c) == len(d['X']) for c, d in zip(cls, ds)) else None
    return X, Y, cls


def to_input(X):
    return 1 - X[:, None].float() / 255  # ink map -> paper-white image, as BitReader/augment expect


def inner_ok(bits, Y):
    """Fraction of strips whose 176 bits pass the koi's inner RS(4) and give back the true chunk."""
    ok = 0
    for b, y in zip(bits.astype(np.uint8), Y.astype(np.uint8)):
        try:
            ok += bytes(koi._RS.decode(np.packbits(b).tobytes())[0]) == np.packbits(y)[:-4].tobytes()
        except ReedSolomonError:
            pass
    return ok / len(bits)


if __name__ == '__main__':
    A = sys.argv[1:] if len(sys.argv) > 4 else ['/content/koi_train_*.npz', '/content/koi_val_*.npz', '/content/koinn.pt', '30000']
    TRAIN, VAL, OUT, STEPS = A[0], A[1], A[2], int(A[3])
    Xtr, Ytr, _ = load(TRAIN)
    Xva, Yva, Cva = load(VAL)
    nbits = Ytr.shape[1]
    print(f'train {tuple(Xtr.shape)} val {tuple(Xva.shape)} bits {nbits} device {dev} '
          f'{torch.cuda.get_device_name(0) if dev == "cuda" else ""}', flush=True)
    if Cva is not None:
        print(f'classical on val: bit acc {(Cva == Yva.cpu().numpy()).mean():.4f}  inner RS ok {inner_ok(Cva, Yva.cpu().numpy()):.3f}',
              flush=True)
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

    def evaluate():
        """Val strips exactly as photographed (no extra augmentation): bit accuracy and inner-RS success."""
        net.eval()
        pred = []
        with torch.no_grad(), torch.autocast(**amp):
            for i in range(0, len(Xva), 128):
                pred.append(net(to_input(Xva[i:i + 128])) > 0)
        net.train()
        pred = torch.cat(pred).cpu().numpy()
        y = Yva.cpu().numpy()
        return (pred == y).mean(), inner_ok(pred, y)

    for step in range(start, STEPS + 1):
        s = AUG * min(1., step / (.3 * STEPS)) * float(torch.rand(1))
        idx = torch.randint(0, len(Xtr), (BS,), device=dev)
        x = augment(to_input(Xtr[idx]), s)
        flip = torch.rand(BS, device=dev) < .5
        x = torch.where(flip[:, None, None, None], x.flip(2, 3), x)
        with torch.autocast(**amp):
            loss = F.binary_cross_entropy_with_logits(net(x).float(), Ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 200:
            acc, ok = evaluate()
            print(f'step {step:6d}  loss {loss.item():.4f}  val bit acc {acc:.4f}  inner RS ok {ok:.3f}  '
                  f'{(time.time() - t0) / (step - start + 1):.3f}s/step', flush=True)
        if step % 2000 == 0 or step == STEPS:
            torch.save({'state': net.state_dict(), 'nbits': nbits, 'step': step,
                        'opt': opt.state_dict(), 'sched': sched.state_dict()}, OUT)
    print('done', flush=True)
