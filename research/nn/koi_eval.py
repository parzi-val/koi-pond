"""Finish and score a koi strip reader checkpoint.

Compiled on the TPU, BatchNorm's running statistics never get written back (the buffers stay at
their initial 0/1), so eval mode would normalise with nonsense. This recomputes them eagerly from
a few thousand strips (cumulative average over train-mode forward passes), then scores the reader
on the validation strips against the classical reader's bits stored there, and writes a
deployable checkpoint (weights + fixed statistics only).

python koi_eval.py CKPT CALIB_GLOB VAL_GLOB [OUT]
"""
import sys, os, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_koi import BitReader, Strips, load, to_input, inner_ok  # noqa: E402


def recalibrate(net, X, n=4096, bs=64):
    """BatchNorm running stats <- exact averages over n strips."""
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average instead of an EMA
    net.train()
    idx = np.random.default_rng(0).permutation(len(X))[:n]
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            net(to_input(torch.from_numpy(X[idx[i:i + bs]])[:, None]))
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = .1
    net.eval()


def predict(net, X, bs=64):
    with torch.no_grad():
        return np.concatenate([(net(to_input(torch.from_numpy(X[i:i + bs])[:, None])) > 0).numpy()
                               for i in range(0, len(X), bs)]).astype(np.uint8)


if __name__ == '__main__':
    ckpt, calib, val = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else None
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    net = BitReader(ck['nbits'])
    net.load_state_dict(ck['state'])
    torch.set_num_threads(os.cpu_count() or 4)
    Xc = load(calib)[0]
    Xva, Yva, sev, _, cls = load(val)
    t = time.time()
    recalibrate(net, Xc)
    pred = predict(net, Xva)
    print(f'step {ck["step"]}: reader bit acc {(pred == Yva).mean():.4f}  inner RS ok {inner_ok(pred, Yva):.3f}  '
          f'({time.time() - t:.0f}s on {len(Xva)} val strips)')
    if cls is not None:
        print(f'classical: bit acc {(cls == Yva).mean():.4f}  inner RS ok {inner_ok(cls, Yva):.3f}')
        for lo, hi in ((0, .4), (.4, .7), (.7, 1.01)):  # by how hard the photo was
            m = (sev >= lo) & (sev < hi)
            print(f'  severity {lo:.1f}-{min(hi, 1):.1f} ({m.sum():4d} strips): reader RS ok {inner_ok(pred[m], Yva[m]):.3f}'
                  f'  classical {inner_ok(cls[m], Yva[m]):.3f}')
    if out:
        torch.save({'state': net.state_dict(), 'nbits': ck['nbits'], 'step': ck['step']}, out)
        print('wrote', out)
