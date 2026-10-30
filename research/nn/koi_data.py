"""Koi strip dataset for the learned pond reader.

Each pond is rendered once, then photographed VIEWS times through a random known warp (rotation,
scale, perspective, crop) and photometric damage (lighting, blur, noise, JPEG). On every view the
classical front end runs exactly as at decode time (koi masks -> spine -> head end), and each koi
it finds is unbent into a SH x SW strip of the ink map: u (nose -> tail) along the width,
v in [-V_SPAN, V_SPAN] body half-widths down the height. The label is the true 176 bits of
whichever koi that was, found by mapping the true nose/tail through the warp, so a strip whose
head end was picked wrong (rev=1) still carries its true bits.

python koi_data.py OUT_PREFIX N_PONDS [SEED] [VIEWS] [--classical]
  -> OUT_PREFIX_000.npz ...  X uint8 (N,SH,SW), Y packed bits (N,22), f, rev, sev, cls (classical
     bits per strip when --classical, for comparison), views (per view: koi in frame, found, matched)
"""
import sys, os, time
from multiprocessing import Pool
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import koi, harness  # noqa: E402

SH, SW, V_SPAN = 128, 640, 1.15
SHARD = 450  # strips per file (~28 MB: under the Colab upload limit)


def strip(ink, fr):
    """Unbend one koi: frame (from koi._frame) -> SH x SW samples of the ink map."""
    sp, nrm, L, us, mid, half = fr
    U, V = np.linspace(0, 1, SW), np.linspace(-V_SPAN, V_SPAN, SH)[:, None]
    T = np.interp(U, us, mid) + V * np.interp(U, us, half)
    X, Y = (np.interp(U, us, sp[:, i]) + T * np.interp(U, us, nrm[:, i]) for i in (0, 1))
    return cv2.remap(ink, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)


def front_end(img, ring=False):
    """The classical decoder up to the read: blurred ink map and one head-first frame per koi found.
    ring: img is an unwarped pond (koi._rectify), so find the koi on their ring."""
    ink = koi._ink(img)
    masks = koi._pond_masks(ink) if ring else koi._koi_masks(ink)
    ink = cv2.GaussianBlur(ink, (0, 0), max(.7, .05 * koi.CELLS[0, 2] * np.mean(koi.L_RNG) * np.sqrt(ink.size) / koi.S))
    frames = []
    for m in masks:
        sp = koi._spine(m)
        if sp is None:
            continue
        m = m.astype(np.float32)
        frames.append(max((koi._frame(m, sp), koi._frame(m, sp[::-1])),
                          key=lambda f: f[5][10:41].mean() - f[5][160:191].mean()))
    return ink, frames


def view(img, rng):
    """A random 'photo' of the pond: known homography H (canvas -> view), then photometric damage."""
    S = img.shape[0]
    sev = rng.uniform(0, 1) ** .7  # skew towards harder views, keep some easy ones
    side = int(S * rng.uniform(1 - .7 * sev, 1))  # output resolution: 0.3 .. 1 of the canvas
    th = rng.uniform(0, 2 * np.pi)
    zoom = rng.uniform(.95, 1 + .3 * sev)  # > 1 crops into the pond
    src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
    c = (src - S / 2) @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    dst = c * zoom * side / S + side / 2
    dst += rng.uniform(-1, 1, (4, 2)) * .18 * sev * side + rng.uniform(-1, 1, 2) * .1 * sev * side
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    out = cv2.warpPerspective(img, H, (side, side), flags=cv2.INTER_AREA if side < S else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    out = harness._lighting(out, rng.uniform(0, .7 * sev), rng)
    if rng.uniform() < .8:
        out = harness._blur(out, rng.uniform(.01, 3.5 * sev + .3), rng)
    out = harness._noise(out, rng.uniform(0, 30 * sev), rng)
    out = harness._jpeg(out, rng.uniform(95 - 85 * sev, 95), rng)
    return out, H, sev


def pond(args):
    i, seed, views, classical = args
    rng = np.random.default_rng((seed, i))
    bits = rng.integers(0, 2, koi.NF * koi.CHUNK * 8).astype(np.uint8)
    z = (seed, i, 0)
    img = koi._render(bits, z)
    words = koi._words(bits)
    kois = koi._layout(np.random.default_rng(z))  # same first draws as the render: true geometry
    ends = np.float32([[k['xy'](0, 0), k['xy'](1, 0)] for k in kois])  # (NF, 2, 2) nose, tail base
    X, Y, F, R, SEV, CLS, V = [], [], [], [], [], [], []
    for _ in range(views):
        im, H, sev = view(img, rng)
        ink, frames = front_end(im)
        e = cv2.perspectiveTransform(ends.reshape(-1, 1, 2), H).reshape(koi.NF, 2, 2)
        inside = ((e >= 0) & (e < im.shape[0])).all((1, 2)).sum()
        matched = 0
        for fr in frames:
            h, t = fr[0][0], fr[0][-1]
            fwd = np.linalg.norm(e[:, 0] - h, axis=1) + np.linalg.norm(e[:, 1] - t, axis=1)
            bwd = np.linalg.norm(e[:, 0] - t, axis=1) + np.linalg.norm(e[:, 1] - h, axis=1)
            f = int(np.argmin(np.minimum(fwd, bwd)))
            if min(fwd[f], bwd[f]) > .5 * fr[2]:  # nowhere near any true koi: clutter or a merged blob
                continue
            matched += 1
            X.append((strip(ink, fr) * 255).round().astype(np.uint8))
            Y.append(np.packbits(words[f]))
            F.append(f)
            R.append(int(bwd[f] < fwd[f]))
            SEV.append(sev)
            if classical:
                CLS.append(np.packbits(classical_read(ink, fr)))
        V.append((inside, len(frames), matched))
    return X, Y, F, R, SEV, CLS, V


def classical_read(ink, fr):
    """koi._read_koi's tracking + hatch read on a given frame -> the 176 bits as seen (no RS)."""
    off = np.zeros((koi.COLS, 4))
    for step in (1, .4):
        off = np.array([o + koi._COMBOS[np.argmax(koi._sample(ink, fr, koi.CELLS[a:b], o + step * koi._COMBOS[:, None], koi.ARC)
                                                  .mean((-1, -2)))] * step if b > a else o
                        for o, a, b in zip(off, koi._STARTS[:-1], koi._STARTS[1:])])
        off = koi.median_filter(off, size=(3, 1), mode='nearest')
    r = koi._sample(ink, fr, koi.CELLS, off[koi._COL], koi.HATCH)
    return (r[:, 0] > r[:, 1]).astype(np.uint8)[:koi.FISH_BITS]


if __name__ == '__main__':
    pos = [a for a in sys.argv[1:] if not a.startswith('--')]
    out, n = pos[0], int(pos[1])
    seed = int(pos[2]) if len(pos) > 2 else 0
    views = int(pos[3]) if len(pos) > 3 else 4
    classical = '--classical' in sys.argv
    t0, buf, k, vstats, total = time.time(), [[] for _ in range(6)], 0, [], []

    def flush():
        global buf, k, vstats
        if buf[0]:
            np.savez_compressed(f'{out}_{k:03d}.npz', X=np.stack(buf[0]), Y=np.stack(buf[1]), f=np.array(buf[2], np.uint8),
                                rev=np.array(buf[3], np.uint8), sev=np.array(buf[4], np.float32),
                                cls=np.stack(buf[5]) if buf[5] else np.zeros((0, 22), np.uint8), views=np.array(vstats))
            k += 1
        buf, vstats = [[] for _ in range(6)], []

    with Pool() as pool:
        for j, res in enumerate(pool.imap_unordered(pond, [(i, seed, views, classical) for i in range(n)])):
            for b, r in zip(buf, res[:6]):
                b += r
            vstats += res[6]
            total += res[6]
            if len(buf[0]) >= SHARD:
                flush()
            if (j + 1) % 20 == 0 or j + 1 == n:
                v = np.array(total)
                print(f'{j + 1}/{n} ponds  {time.time() - t0:.0f}s  koi in frame {v[:, 0].sum()} found {v[:, 1].sum()} '
                      f'matched {v[:, 2].sum()}', flush=True)
    flush()
