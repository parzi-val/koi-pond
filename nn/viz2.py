"""What does the v2 bit-query model look at? Its own cross-attention, per bit, vs the true scale.

python nn/viz2.py OUT_DIR [weights] [n_plates]
Writes attention.png (8 bits, attention heat + true scale in green), attn_all.png (every bit's
attention peak joined to its true scale; green = inside its own scale), distort.png (the same
bits under heavy distortion) and prints the hit rate.
"""
import sys, os
import numpy as np, cv2, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fish, harness  # noqa: E402
from train2 import BitReader  # noqa: E402

OUT = sys.argv[1]
ck = torch.load(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), 'fishnn2.pt'),
                map_location='cpu', weights_only=False)
net = BitReader(ck['nbits'])
net.load_state_dict(ck['state'])
net.eval()
W_, H_ = 640, 400
gray = lambda img: cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (W_, H_), interpolation=cv2.INTER_AREA)


def run(g):
    with torch.no_grad():
        lg, at = net(torch.from_numpy(g).float()[None, None] / 255, attn=True)
    return (lg[0] > 0).numpy().astype(np.uint8), at[0].numpy()


def heat(a):
    a = cv2.resize(a, (W_, H_), interpolation=cv2.INTER_CUBIC)
    return np.clip(a / (a.max() + 1e-9), 0, 1)


rng = np.random.default_rng(42)
bits = rng.integers(0, 2, fish.CODE_BITS).astype(np.uint8)
style = (777, 0)
img = fish._render(bits, style)
g = gray(img)
pred, att = run(g)
print(f'bit accuracy on this plate: {(pred == bits).mean():.3f}')


def true_cell(k):
    b2 = bits.copy()
    b2[k] ^= 1
    d = np.abs(gray(fish._render(b2, style)).astype(np.float32) - g.astype(np.float32))
    return d / (d.max() + 1e-6)


def panel(base_gray, a, t, label):
    p = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR).astype(np.float32) * .55
    p += cv2.applyColorMap((heat(a) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32) * .6
    if t is not None:
        p[t > .2] = (60, 255, 60)
    p = np.clip(p, 0, 255).astype(np.uint8)
    cv2.putText(p, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    return p


hits, panels, where = 0, [], cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
show = list(np.linspace(0, fish.CODE_BITS - 1, 8).astype(int))
cells = {}
for k in range(fish.CODE_BITS):
    t = true_cell(k)
    cells[k] = t
    a = heat(att[k])
    py, px = np.unravel_index(np.argmax(a), a.shape)
    ty, tx = np.unravel_index(np.argmax(cv2.GaussianBlur(t, (0, 0), 2)), t.shape)
    hit = cv2.dilate((t > .2).astype(np.uint8), np.ones((11, 11), np.uint8))[py, px] > 0
    hits += hit
    cv2.line(where, (tx, ty), (px, py), (0, 180, 0) if hit else (0, 0, 255), 1)
    cv2.circle(where, (tx, ty), 2, (255, 120, 0), -1)
    if k in show:
        panels.append(panel(g, att[k], t, f'bit {k}  pred {pred[k]} true {bits[k]}'))
print(f'attention peak inside its own scale: {hits}/{fish.CODE_BITS} bits')
cv2.imwrite(os.path.join(OUT, 'attention.png'), np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, 8, 2)]))
cv2.imwrite(os.path.join(OUT, 'attn_all.png'), where)

# the same plate photographed badly: does each bit still find its scale?
d = harness.DISTORTIONS['photo'][0](img, 2, np.random.default_rng(3))
gd = gray(d)
pd, ad = run(gd)
print(f'bit accuracy under photo-2 distortion: {(pd == bits).mean():.3f}')
cv2.imwrite(os.path.join(OUT, 'distort.png'), np.vstack([np.hstack([panel(gd, ad[k], None, f'bit {k} (distorted)')
                                                                      for k in show[i:i + 2]]) for i in range(0, 8, 2)]))
print('wrote', OUT)
