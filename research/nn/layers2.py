"""Which v2 decoder layer does the reading? Per-layer cross-attention vs the true scale.

python nn/layers2.py OUT_DIR   -> layers.png (4 bits x 3 layers) + per-layer hit rates
"""
import sys, os
import numpy as np, cv2, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fish  # noqa: E402
from train2 import BitReader, pos2d  # noqa: E402

OUT = sys.argv[1]
ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fishnn2.pt'), map_location='cpu', weights_only=False)
net = BitReader(ck['nbits'])
net.load_state_dict(ck['state'])
net.eval()
gray = lambda img: cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (640, 400), interpolation=cv2.INTER_AREA)

rng = np.random.default_rng(42)
bits = rng.integers(0, 2, fish.CODE_BITS).astype(np.uint8)
style = (777, 0)
g = gray(fish._render(bits, style))

with torch.no_grad():  # the forward pass, keeping every layer's cross-attention
    x = torch.from_numpy(g).float()[None, None] / 255
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, 400), torch.linspace(-1, 1, 640), indexing='ij')
    f = net.backbone(torch.cat([1 - x, torch.stack([xs, ys])[None]], 1))
    h, w = f.shape[2:]
    mem = f.flatten(2).transpose(1, 2)
    mpos = pos2d(h, w, mem.shape[2], 'cpu')[None]
    qpos = net.qpos[None]
    q = torch.zeros_like(qpos)
    atts = []
    for layer in net.dec:
        q, wts = layer(q, qpos, mem, mpos, keep=True)
        atts.append(wts[0].view(-1, h, w).numpy())


def true_cell(k):
    b2 = bits.copy()
    b2[k] ^= 1
    d = np.abs(gray(fish._render(b2, style)).astype(np.float32) - g.astype(np.float32))
    return d / (d.max() + 1e-6)


up = lambda a: cv2.resize(a, (640, 400), interpolation=cv2.INTER_CUBIC)
hits = np.zeros(3, int)
show = [0, 70, 141, 211]
rows = []
for k in range(fish.CODE_BITS):
    t = true_cell(k)
    near = cv2.dilate((t > .2).astype(np.uint8), np.ones((11, 11), np.uint8))
    row = []
    for li in range(3):
        a = up(atts[li][k])
        py, px = np.unravel_index(np.argmax(a), a.shape)
        hits[li] += near[py, px] > 0
        if k in show:
            p = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32) * .55
            p += cv2.applyColorMap((np.clip(a / (a.max() + 1e-9), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO) * .6
            p[t > .2] = (60, 255, 60)
            p = np.clip(p, 0, 255).astype(np.uint8)
            cv2.putText(p, f'bit {k}  layer {li + 1}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            row.append(cv2.resize(p, (480, 300)))
    if row:
        rows.append(np.hstack(row))
for li in range(3):
    print(f'layer {li + 1}: attention peak inside own scale for {hits[li]}/{fish.CODE_BITS} bits')
cv2.imwrite(os.path.join(OUT, 'layers.png'), np.vstack(rows))
