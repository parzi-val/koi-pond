"""Corner anchors vs the plain front end, on simulated phone photos: the whole pond in frame at a random
size, angle and perspective, on a cluttered background (screens, text, edges), then blur, noise, lighting, JPEG.

python anchor_bench.py [N_PONDS] [VIEWS]
"""
import sys, os
from multiprocessing import Pool
import numpy as np, cv2

W, H = 1500, 2000  # portrait phone frame (the app reads 2250x3000; smaller keeps this quick)


def clutter(rng):
    """A dark, busy backdrop: panels, UI-ish text, hard edges."""
    bg = np.full((H, W, 3), int(rng.uniform(20, 90)), np.uint8)
    for _ in range(rng.integers(10, 25)):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.rectangle(bg, (int(x), int(y)), (int(x + rng.integers(50, 700)), int(y + rng.integers(20, 400))),
                      rng.uniform(0, 255, 3).tolist(), -1 if rng.uniform() < .7 else int(rng.integers(2, 8)))
    for _ in range(rng.integers(20, 60)):
        cv2.putText(bg, ''.join(rng.choice(list('abcdefghij koi pond 0123')) for _ in range(rng.integers(5, 30))),
                    (int(rng.integers(0, W)), int(rng.integers(0, H))), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(.5, 2),
                    rng.uniform(0, 255, 3).tolist(), int(rng.integers(1, 4)))
    return bg.astype(np.float32)


def photo(img, rng):
    import koi, harness
    S = img.shape[0]
    sev = rng.uniform(0, 1) ** .7
    for _ in range(100):
        frac, th = rng.uniform(.35, .9), rng.uniform(0, 2 * np.pi)
        side = frac * W
        src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
        c = (src - S / 2) / S * side @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        dst = c + rng.uniform(-.1, .1, (4, 2)) * sev * side + [W / 2 + rng.uniform(-.2, .2) * W, H / 2 + rng.uniform(-.2, .2) * H]
        if dst.min() > 5 and dst[:, 0].max() < W - 5 and dst[:, 1].max() < H - 5:
            break
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    pond = cv2.warpPerspective(img.astype(np.float32), M, (W, H), flags=cv2.INTER_AREA if side < S else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((S, S), np.float32), M, (W, H))[..., None]
    out = (pond * mask + clutter(rng) * (1 - mask)).clip(0, 255).astype(np.uint8)
    out = harness._lighting(out, rng.uniform(0, .6 * sev), rng)
    out = harness._blur(out, rng.uniform(.01, 2.5 * sev + .3), rng)
    out = harness._noise(out, rng.uniform(0, 25 * sev), rng)
    out = harness._jpeg(out, rng.uniform(95 - 75 * sev, 95), rng)
    return out, sev, frac


def pond(args):
    i, views = args
    import torch
    torch.set_num_threads(1)
    import koi, koi_nn
    rng = np.random.default_rng((11, i))
    bits = rng.integers(0, 2, koi.NF * koi.CHUNK * 8).astype(np.uint8)
    img = koi._render(bits, (11, i, 0))
    out = []
    for _ in range(views):
        im, sev, frac = photo(img, rng)
        row = [sev, frac, koi._anchors(im) is not None]
        for rect in (False, True):
            koi_nn.RECTIFY = rect
            r = koi_nn.decode(im)
            read = 0 if r is None else koi.NF - len(r[1]) // koi.CHUNK
            ok = False
            if r is not None and len(r[1]) <= koi.CHUNK:
                b, e = r
                ok = bool((b.reshape(-1, 8)[[j for j in range(koi.NF * koi.CHUNK) if j not in e]] ==
                           bits.reshape(-1, 8)[[j for j in range(koi.NF * koi.CHUNK) if j not in e]]).all())
            row += [read, ok]
        out.append(row)
    return out


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    views = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool() as p:
        R = np.array(sum(p.map(pond, [(i, views) for i in range(n)]), []), float)
    sev, frac, anch, r0, ok0, r1, ok1 = R.T
    print(f'{len(R)} photos  anchors found {anch.mean():.2f}')
    for name, m in (('all', sev >= 0), ('sev <.4', sev < .4), ('sev .4-.7', (sev >= .4) & (sev < .7)), ('sev >.7', sev >= .7),
                    ('pond <.55 of width', frac < .55), ('pond >.55', frac >= .55)):
        print(f'  {name:20s} n={m.sum():3.0f}  koi read/6: plain {r0[m].mean():.2f}  anchors {r1[m].mean():.2f}   '
              f'message ok: plain {ok0[m].mean():.2f}  anchors {ok1[m].mean():.2f}')
