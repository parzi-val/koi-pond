"""koinn.pt -> koinn.onnx for the Android app: input ink strips (B,1,SH,SW) in [0,1], output bit logits (B,176).
Checks ONNX Runtime against torch on real validation strips.

python export_koi.py [CKPT OUT VAL_GLOB]
"""
import sys, os, glob
import numpy as np, torch, onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train2 import BitReader  # noqa: E402
from koi_data import SH, SW  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
A = sys.argv[1:] if len(sys.argv) > 3 else [f'{here}/koinn.pt', f'{here}/../../android/app/src/main/assets/koinn.onnx',
                                             f'{here}/../data/koi/koi_val_000.npz']
ck = torch.load(A[0], map_location='cpu')
net = BitReader(ck['nbits'])
net.load_state_dict(ck['state'])
net.eval()


class Ink(torch.nn.Module):  # the app hands over ink, not the training inputs' 1 - ink
    def forward(self, ink):
        return net(1 - ink)


os.makedirs(os.path.dirname(A[1]), exist_ok=True)
torch.onnx.export(Ink(), (torch.rand(2, 1, SH, SW),), A[1], input_names=['ink'], output_names=['logits'],
                  dynamic_shapes={'ink': {0: torch.export.Dim('B', max=64)}}, dynamo=True, external_data=False)

X = np.concatenate([np.load(p)['X'] for p in sorted(glob.glob(A[2]))])[:64]
ink = (X[:, None] / 255).astype(np.float32)
with torch.no_grad():
    ref = Ink()(torch.from_numpy(ink)).numpy()
got = ort.InferenceSession(A[1]).run(None, {'ink': ink})[0]
agree = ((ref > 0) == (got > 0)).mean()
print(f'{os.path.getsize(A[1]) / 1e6:.1f} MB  max |dlogit| {np.abs(ref - got).max():.2e}  bit agreement {agree:.5f}')
assert agree > .9995
