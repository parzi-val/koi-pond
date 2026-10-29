"""Render a training set for the learned fish decoder, as shards.

Each sample: a clean fishdraw plate (random bits, random style), grayscale, resized to WIDTH x
WIDTH*5/8, with its CODE_BITS codeword bits as the label. Distortions are NOT baked in; the
trainer augments on GPU. Shards of SHARD samples are written as they finish.

Usage: python nn/gen_data.py N OUT_PREFIX [seed] [width] [shard]
       -> OUT_PREFIX_000.npz, OUT_PREFIX_001.npz, ...   (or OUT_PREFIX.npz when N <= shard)
"""
import sys, os
from multiprocessing import Pool
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import fish  # noqa: E402

SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
WIDTH = int(sys.argv[4]) if len(sys.argv) > 4 else 512
SHARD = int(sys.argv[5]) if len(sys.argv) > 5 else 500


def one(i):
    rng = np.random.default_rng((SEED, i))
    bits = rng.integers(0, 2, fish.CODE_BITS).astype(np.uint8)
    img = fish._render(bits, (SEED, i, 0))
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (WIDTH, WIDTH * 5 // 8), interpolation=cv2.INTER_AREA), np.packbits(bits)


def save(path, res):
    np.savez_compressed(path, X=np.stack([r[0] for r in res]), Y=np.stack([r[1] for r in res]), code_bits=fish.CODE_BITS)
    print('saved', path, f'{os.path.getsize(path) / 1e6:.0f} MB', flush=True)


if __name__ == '__main__':
    n, prefix = int(sys.argv[1]), sys.argv[2]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        buf, k = [], 0
        for r in pool.imap(one, range(n), chunksize=8):
            buf.append(r)
            if len(buf) == SHARD:
                save(prefix + '.npz' if n <= SHARD else f'{prefix}_{k:03d}.npz', buf)
                buf, k = [], k + 1
        if buf:
            save(prefix + '.npz' if n <= SHARD else f'{prefix}_{k:03d}.npz', buf)
