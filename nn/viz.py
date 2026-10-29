"""What did the fish CNN learn? Saliency per bit vs the true scale, first-layer filters, feature maps.

python nn/viz.py OUT_DIR [weights]
Writes saliency.png, where.png, filters.png, features.png and prints how many bits look at their own scale.
"""
import sys, os
import numpy as np, cv2, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fish  # noqa: E402
from nn.train import Net  # noqa: E402

OUT = sys.argv[1]
ck = torch.load(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), 'fishnn.pt'), map_location='cpu')
net = Net(ck['nbits'])
net.load_state_dict(ck['state'])
net.eval()
gray = lambda img: cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (512, 320), interpolation=cv2.INTER_AREA)

rng = np.random.default_rng(42)
bits = rng.integers(0, 2, fish.CODE_BITS).astype(np.uint8)
style = (777, 0)
img = fish._render(bits, style)
g = gray(img)
x = torch.from_numpy(g).float()[None, None].div(255).requires_grad_(True)
logits = net(x)[0]
pred = (logits > 0).numpy().astype(np.uint8)
print(f'bit accuracy on this plate: {(pred == bits).mean():.3f}')


def true_cell(k):
    """Flip bit k, re-render the same style: the pixels that change are exactly scale k's hatches."""
    b2 = bits.copy()
    b2[k] ^= 1
    d = np.abs(gray(fish._render(b2, style)).astype(np.float32) - g.astype(np.float32))
    return d / (d.max() + 1e-6)


def saliency(k):
    net.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    logits[k].backward(retain_graph=True)
    s = cv2.GaussianBlur(np.abs(x.grad[0, 0].numpy()), (0, 0), 1.5)
    return s / (s.max() + 1e-12)


# 1) saliency vs truth for a spread of bits; 2) peak-in-own-scale score over all bits
hits, panels, where = 0, [], cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
show = set(np.linspace(0, fish.CODE_BITS - 1, 8).astype(int))
for k in range(fish.CODE_BITS):
    s, t = saliency(k), true_cell(k)
    py, px = np.unravel_index(np.argmax(s), s.shape)
    ty, tx = np.unravel_index(np.argmax(cv2.GaussianBlur(t, (0, 0), 2)), t.shape)
    hit = cv2.dilate((t > .2).astype(np.uint8), np.ones((9, 9), np.uint8))[py, px] > 0
    hits += hit
    cv2.line(where, (tx, ty), (px, py), (0, 180, 0) if hit else (0, 0, 255), 1)
    cv2.circle(where, (tx, ty), 2, (255, 120, 0), -1)
    if k in show:
        base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32) * .55
        heat = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32)
        p = base + heat * .6
        p[t > .2] = (60, 255, 60)  # true scale k in green
        p = np.clip(p, 0, 255).astype(np.uint8)
        cv2.putText(p, f'bit {k}  pred {pred[k]} true {bits[k]}', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        panels.append(p)
print(f'saliency peak inside its own scale: {hits}/{fish.CODE_BITS} bits')
cv2.imwrite(os.path.join(OUT, 'saliency.png'), np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]))
cv2.imwrite(os.path.join(OUT, 'where.png'), cv2.resize(where, (1024, 640), interpolation=cv2.INTER_NEAREST))

# 3) first-layer filters (ink channel), upscaled
w = net.stem[0].weight.detach()[:, 0].numpy()
tiles = [cv2.resize(((f - f.min()) / (np.ptp(f) + 1e-9) * 255).astype(np.uint8), (60, 60), interpolation=cv2.INTER_NEAREST)
         for f in w]
tiles = [cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255) for t in tiles]
cv2.imwrite(os.path.join(OUT, 'filters.png'), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))

# 4) feature maps after trunk stages 1 and 2: the 8 most active channels of each
acts = {}
hooks = [net.trunk[i].register_forward_hook(lambda m, a, o, i=i: acts.__setitem__(i, o.detach()[0])) for i in (0, 1, 2)]
with torch.no_grad():
    net(x.detach())
for h in hooks:
    h.remove()
rows = []
for i in (0, 1, 2):
    a = acts[i]
    top = torch.argsort(a.flatten(1).std(1), descending=True)[:6]
    row = []
    for c in top:
        m = a[c].numpy()
        m = cv2.resize(((m - m.min()) / (np.ptp(m) + 1e-9) * 255).astype(np.uint8), (256, 160), interpolation=cv2.INTER_NEAREST)
        m = cv2.applyColorMap(m, cv2.COLORMAP_VIRIDIS)
        cv2.putText(m, f'stage {i + 1} ch {int(c)}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        row.append(m)
    rows.append(np.hstack(row))
cv2.imwrite(os.path.join(OUT, 'features.png'), np.vstack(rows))
print('wrote', OUT)
