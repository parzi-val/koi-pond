"""Shared benchmark: payload -> RS codeword -> render -> distort -> decode -> metrics.

A method module exposes:
    encode(bits: uint8[CODE_BITS], z: int) -> HxWx3 uint8 RGB image
    decode(img) -> (bits uint8[CODE_BITS], erased_byte_indices) | None   (None = no code detected)
    optional PAYLOAD_BYTES / PARITY_BYTES to override the default RS(12, 8)

Usage: python harness.py topo --n 40 [--only blur jpeg]
"""
import argparse, csv, importlib, os, time
import numpy as np, cv2
from reedsolo import RSCodec, ReedSolomonError

PAYLOAD_BYTES, PARITY_BYTES = 8, 4  # 64-bit payload, RS(12,8): fixes 2 byte errors or 4 erasures
CODE_BITS = 8 * (PAYLOAD_BYTES + PARITY_BYTES)
_RS = {}  # one codec per parity size; methods may override PAYLOAD_BYTES / PARITY_BYTES


def _rs(nsym):
    return _RS.setdefault(nsym, RSCodec(nsym))


def ecc_encode(payload: bytes, nsym=PARITY_BYTES) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes(_rs(nsym).encode(payload)), np.uint8))


def ecc_decode(bits, erased=(), nsym=PARITY_BYTES):
    try:
        return bytes(_rs(nsym).decode(np.packbits(bits).tobytes(), erase_pos=list(erased) or None)[0])
    except ReedSolomonError:
        return None


# --- distortions: fn(img, level, rng) -> img -------------------------------------------

def _scale(img, s, rng):
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _rotate(img, deg, rng):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg * rng.choice([-1, 1]), 1.0)
    nw, nh = int(h * abs(M[0, 1]) + w * abs(M[0, 0])), int(h * abs(M[0, 0]) + w * abs(M[0, 1]))
    M[:, 2] += (nw - w) / 2, (nh - h) / 2
    return cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


def _jpeg(img, q, rng):
    _, buf = cv2.imencode('.jpg', img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


def _blur(img, sigma, rng):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def _noise(img, sigma, rng):
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _perspective(img, f, rng):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + (rng.uniform(0, f, (4, 2)) * [[1, 1], [-1, 1], [-1, -1], [1, -1]] * [w, h]).astype(np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)


def _lighting(img, a, rng):
    h, w = img.shape[:2]
    t = rng.uniform(0, 2 * np.pi)
    y, x = np.mgrid[0:h, 0:w]
    ramp = np.clip(((x / w - .5) * np.cos(t) + (y / h - .5) * np.sin(t)) * 2, -1, 1)
    gain = (1 - a * (.5 + .5 * ramp))[..., None] * (1 + .3 * rng.uniform(-a, a, 3))  # shadow + colour cast
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def _crop(img, f, rng):
    h, w = img.shape[:2]
    l, t = rng.uniform(0, f, 2)
    return img[int(t * h):h - int((f - t) * h), int(l * w):w - int((f - l) * w)]


def _photo(img, lv, rng):  # rough phone-photo chain
    for fn, v in [(_perspective, .04 * lv), (_rotate, 10 * lv), (_lighting, .2 * lv), (_scale, 1 - .15 * lv),
                  (_blur, .4 * lv), (_noise, 2.5 * lv), (_jpeg, 90 - 12 * lv)]:
        img = fn(img, v, rng)
    return img


DISTORTIONS = {
    'none': (lambda im, v, r: im, [0]),
    'scale': (_scale, [.75, .5, .35]),
    'rotate': (_rotate, [15, 45, 135]),
    'jpeg': (_jpeg, [75, 50, 25, 10]),
    'blur': (_blur, [1, 2, 3]),
    'noise': (_noise, [8, 16, 32]),
    'perspective': (_perspective, [.05, .1, .2]),
    'lighting': (_lighting, [.3, .5, .7]),
    'crop': (_crop, [.05, .1, .2]),
    'photo': (_photo, [1, 2, 3]),
}


def save(path, img):
    cv2.imwrite(path, np.ascontiguousarray(img[..., ::-1]))


def run(method, n, seed=0, only=None):
    m = importlib.import_module(method)
    out = os.path.join('out', method)
    os.makedirs(os.path.join(out, 'samples'), exist_ok=True)
    pb, nsym = getattr(m, 'PAYLOAD_BYTES', PAYLOAD_BYTES), getattr(m, 'PARITY_BYTES', PARITY_BYTES)
    print(f'{method}: {pb}-byte payload + {nsym} parity bytes')
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        payload = rng.bytes(pb)
        code = ecc_encode(payload, nsym)
        items.append((payload, code, m.encode(code, z=seed * 100_000 + i)))
    thumbs = [cv2.resize(im, (256, 256 * im.shape[0] // im.shape[1]), interpolation=cv2.INTER_AREA) for _, _, im in items[:12]]
    thumbs += [np.full_like(thumbs[0], 255)] * (-len(thumbs) % 4)
    save(os.path.join(out, 'contact_sheet.png'), np.vstack([np.hstack(thumbs[r:r + 4]) for r in range(0, len(thumbs), 4)]))

    rows = []
    print(f"{'distortion':<12}{'level':>7}{'detect':>8}{'BER':>8}{'exact':>8}{'ms':>8}")
    for di, (name, (fn, levels)) in enumerate(DISTORTIONS.items()):
        if only and name not in only:
            continue
        for lv in levels:
            det = ok = 0
            ber, t = [], 0.0
            for i, (payload, code, img) in enumerate(items):
                d = fn(img, lv, np.random.default_rng((seed, di, i)))
                if i < 2:
                    save(os.path.join(out, 'samples', f'{name}_{lv}_{i}.png'), d)
                t0 = time.perf_counter()
                res = m.decode(d)
                t += time.perf_counter() - t0
                if res is None:
                    continue
                bits, erased = res
                det += 1
                ber.append(np.mean(bits != code))
                ok += ecc_decode(bits, erased, nsym) == payload
            row = dict(distortion=name, level=lv, detect=det / n, ber=np.mean(ber) if ber else float('nan'),
                       exact=ok / n, ms=1000 * t / n)
            rows.append(row)
            print(f"{name:<12}{lv:>7}{row['detect']:>8.2f}{row['ber']:>8.3f}{row['exact']:>8.2f}{row['ms']:>8.0f}")
    with open(os.path.join(out, 'results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('method')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    run(a.method, a.n, a.seed, a.only)
