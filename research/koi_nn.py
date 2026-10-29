"""Koi pond with the learned strip reader: classical front end (koi masks -> spine -> frame), each koi
unbent into a strip (nn/koi_data.py) and read by nn/koinn.pt, then the usual inner/outer RS.
Renders are koi.py's, so this benchmarks against koi.py on the very same ponds.
"""
import os
import numpy as np, torch
import koi
from koi import encode, PAYLOAD_BYTES, PARITY_BYTES  # noqa: F401  (harness API)
from nn.koi_data import front_end, strip
from nn.train2 import BitReader
from reedsolo import ReedSolomonError

_NET = None


def _net():
    global _NET
    if _NET is None:
        ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn', 'koinn.pt'), map_location='cpu')
        _NET = BitReader(ck['nbits'])
        _NET.load_state_dict(ck['state'])
        _NET.eval()
    return _NET


def decode(img):
    ink, frames = front_end(img)
    if not frames:
        return None
    x = 1 - torch.from_numpy(np.stack([strip(ink, fr) for fr in frames]))[:, None].float()
    with torch.no_grad():  # as photographed, and v-flipped for a mirrored photo
        bits = (torch.cat([_net()(x), _net()(x.flip(2))]) > 0).numpy().astype(np.uint8)
    chunks = {}
    for b in bits:
        try:
            msg = bytes(koi._RS.decode(np.packbits(b).tobytes())[0])
        except ReedSolomonError:
            continue
        if msg[0] >> 4 == 0xA and (msg[0] & 15) < koi.NF:
            chunks.setdefault(msg[0] & 15, np.frombuffer(msg[1:], np.uint8))
    if not chunks:
        return None
    out, erased = np.zeros(koi.NF * koi.CHUNK, np.uint8), []
    for f in range(koi.NF):
        if f in chunks:
            out[f::koi.NF] = chunks[f]
        else:
            erased += list(range(f, koi.NF * koi.CHUNK, koi.NF))
    return np.unpackbits(out), erased
