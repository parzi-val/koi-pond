"""Learned decoder v2 for the fishdraw plates: CNN feature grid + one attention query per bit.

Same encoder as fish.py; the network (nn/train2.py) reads a 400x640 grayscale view of the plate
and outputs every codeword bit directly.
"""
import os
import numpy as np, cv2, torch
from fish import encode, PAYLOAD_BYTES, PARITY_BYTES, CODE_BITS  # noqa: F401  (harness API)
from nn.train2 import BitReader

_NET = None


def _net():
    global _NET
    if _NET is None:
        ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn', 'fishnn2.pt'),
                        map_location='cpu', weights_only=False)
        _NET = BitReader(ck['nbits'])
        _NET.load_state_dict(ck['state'])
        _NET.eval()
    return _NET


def decode(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (640, 400), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        logits = _net()(torch.from_numpy(g).float()[None, None] / 255)[0]
    return (logits > 0).numpy().astype(np.uint8), []
