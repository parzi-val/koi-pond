"""Learned decoder for the fishdraw plates (same encoder as fish.py), for the shared harness.

The CNN (nn/train.py) reads a 320x512 grayscale view of the whole plate and outputs every
codeword bit directly: no silhouette, no grid, no tracking.
"""
import os
import numpy as np, cv2, torch
import fish
from fish import encode, PAYLOAD_BYTES, PARITY_BYTES, CODE_BITS  # noqa: F401  (harness API)
from nn.train import Net

_NET = None


def _net():
    global _NET
    if _NET is None:
        ck = torch.load(os.path.join(os.path.dirname(__file__), 'nn', 'fishnn.pt'), map_location='cpu')
        _NET = Net(ck['nbits'])
        _NET.load_state_dict(ck['state'])
        _NET.eval()
    return _NET


def decode(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (512, 320), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        logits = _net()(torch.from_numpy(g).float()[None, None] / 255)[0]
    return (logits > 0).numpy().astype(np.uint8), []
