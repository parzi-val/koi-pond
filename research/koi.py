"""Koi pond: a swirl of top-down koi, each carrying a chunk of the payload in its scale hatches.

Each koi is a curved version of fish.py's body: scales on a (u, v) grid laid along a bent spine
(u: nose 0 -> tail base 1, v: across the back -1 .. +1); the bit is which side of each scale its
hatch group sits on. The bold outline is the sync mark, as in fish.py. Fins, plants and ripples
are thin, and every wash (water, koi patches, lily pads) is a pale, high-value colour, so it
vanishes in the value channel (max of R, G, B) the decoder reads.

Payload: the harness's RS codeword (sized so one whole koi can be lost) is dealt round-robin
across NF koi. Each koi carries [0xA0 | index] + chunk + 4 inner RS bytes; the inner code tells
the decoder which way round a koi is (mirror) and which chunk it holds, and a koi that fails it
becomes a block of erasures for the outer code.
"""
import numpy as np, cv2
from PIL import Image, ImageDraw
from reedsolo import RSCodec, ReedSolomonError
from scipy.ndimage import median_filter, gaussian_filter1d, maximum_filter, minimum_filter, uniform_filter1d
from fish import _wobble, _smooth, _odd, _COMBOS, ARC, HATCH

S, SS = 2400, 2
PAPER, INK = np.array([255., 250., 240.]), np.array([24., 20., 18.])
BOLD, THIN = 3.6, 1.3
NF = 6                         # koi in the pond
L_RNG, HW_RNG, P_RNG = (640, 700), (90, 110), (16, 22)
# corner anchors: a bold ring + centre dot (a stepping stone with a carved bullseye) in each corner, like a QR
# code's finder patterns; the decoder maps them to these points and unwarps the pond before looking for koi
ANCHOR_M, ANCHOR_R, ANCHOR_W, ANCHOR_DOT = 125, 80, 20, 26
ANCHORS = np.float32([[ANCHOR_M, ANCHOR_M], [S - ANCHOR_M, ANCHOR_M], [S - ANCHOR_M, S - ANCHOR_M], [ANCHOR_M, S - ANCHOR_M]])
RECT = 1600  # side of the unwarped pond the decoder reads
C_RNG, D_RNG, K_RNG = (.6, .75), (.6, .85), (.02, .04)
COLS, U0, U1 = 24, .22, .88    # coded scale columns span body fraction U0..U1
VM, DV_PX = .86, 15            # scales cover v in [-VM, VM]; minimum scale height in px


def _profile(u, Hw, c, d, P, k):
    """Half-width of a koi seen from above: broad shoulders, rounded snout (k), tail stalk P."""
    u = np.clip(u, 0, 1)
    return Hw * np.sin(np.pi * u ** c) ** d * (1 - np.exp(-u / k)) + P * u ** 2


def _min_half(u):
    g = np.linspace(0, 1, 5)
    c, d = np.meshgrid(C_RNG[0] + g * np.ptp(C_RNG), D_RNG[0] + g * np.ptp(D_RNG))
    return min(_profile(u, HW_RNG[0], ci, di, P_RNG[0], K_RNG[1]) for ci, di in zip(c.ravel(), d.ravel()))


def _cells(cols):
    """(u, v, du, dv, col) in bit order. Odd columns have one row fewer, offset half a row, so the
    layout is symmetric in v: a mirrored koi lands on the same grid."""
    du = (U1 - U0) / COLS
    out = []
    for c in cols:
        u = U0 + (c + .5) * du
        n0 = max(2, int(2 * VM * _min_half(u) / DV_PX))
        dv, n = 2 * VM / n0, n0 - c % 2
        out += [(u, (j - (n - 1) / 2) * dv, du, dv, c) for j in range(n)]
    return np.array(out)


CELLS = _cells(range(COLS))
EXTRA = _cells(range(COLS, COLS + 2))  # decorative scales toward the tail, not read
FISH_BITS = len(CELLS) // 8 * 8
CHUNK = FISH_BITS // 8 - 5            # per-koi bytes after the index byte and 4 inner parity bytes
PAYLOAD_BYTES, PARITY_BYTES = (NF - 1) * CHUNK, CHUNK  # outer code survives losing any one koi
_RS = RSCodec(4)
_COL = CELLS[:, 4].astype(int)
_STARTS = np.searchsorted(_COL, np.arange(COLS + 1))
_KEY = {(int(c), round(v, 6)): i for i, (u, v, du, dv, c) in enumerate(CELLS)}
MIRROR = np.array([_KEY[(int(c), round(-v, 6))] for u, v, du, dv, c in CELLS])  # cell <-> its mirror twin


def _hsv(h, s, v):
    import colorsys
    return np.array(colorsys.hsv_to_rgb(h % 1, s, v)) * 255


def _koi(rng, ctr, R, phi, spin):
    """One koi swimming head-first around the ring: geometry closures in body coordinates."""
    L, Hw, P = rng.uniform(*L_RNG), rng.uniform(*HW_RNG), rng.uniform(*P_RNG)
    c, d, k = rng.uniform(*C_RNG), rng.uniform(*D_RNG), rng.uniform(*K_RNG)
    s = np.linspace(0, L, 600)
    A, lam, ph = rng.uniform(0, 12), rng.uniform(.8, 1.3) * L, rng.uniform(0, 2 * np.pi)  # gentle sway
    al = phi + spin * (L / 2 - s) / R
    raw = ctr + (R + A * np.sin(2 * np.pi * s / lam + ph))[:, None] * np.stack([np.cos(al), np.sin(al)], -1)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))])
    us = arc / arc[-1]
    tg = np.gradient(raw, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], -1)
    half = lambda u: _profile(u, Hw, c, d, P, k)
    at = lambda a, u: np.stack([np.interp(u, us, a[:, i]) for i in (0, 1)], -1)

    def xy(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        return at(raw, u) + at(nrm, u) * (v * half(u))[..., None]

    uu = np.linspace(0, 1, 300)
    return dict(xy=xy, half=half, L=arc[-1], tan=lambda u: at(tg, u), nrm=lambda u: at(nrm, u),
                poly=np.vstack([xy(uu, -1), xy(uu[::-1], 1)]))


def encode(bits, z):
    """Render, then self-verify; re-roll the style (never the bits) if the clean render misreads."""
    for attempt in range(6):
        img = _render(bits, (z, attempt))
        res = decode(img)
        if res is not None and not res[1] and (res[0] == bits).all():
            return img
    return img  # ponytail: gives up after 6 styles; shows up as a clean-decode failure in the harness


def _words(bits):
    """Outer codeword bits -> each koi's 176 bits: [0xA0 | index] + chunk + 4 inner parity bytes."""
    code = np.packbits(bits)
    return [np.unpackbits(np.frombuffer(bytes(_RS.encode(bytes([0xA0 | f]) + code[f::NF].tobytes())), np.uint8))
            for f in range(NF)]


def _layout(rng):
    """Koi swim a ring head-first, in shuffled order; kois[f] carries chunk f. First draws from a render's rng."""
    ctr = np.full(2, S / 2)
    R0, phase, spin = rng.uniform(870, 910), rng.uniform(0, 2 * np.pi), rng.choice([-1, 1])
    slots = rng.permutation(NF)
    return [_koi(rng, ctr, R0 + rng.uniform(-25, 25), phase + 2 * np.pi * slots[f] / NF + rng.uniform(-.05, .05), spin)
            for f in range(NF)]


def _render(bits, z):
    rng = np.random.default_rng(z)
    words = _words(bits)
    ctr, kois = np.full(2, S / 2), _layout(rng)

    # --- ink layer: supersampled alpha, composited over the washes at the end ------------
    A = Image.new('L', (S * SS, S * SS), 0)
    dr = ImageDraw.Draw(A)

    def line(pts, w, a=255):
        dr.line([tuple(p) for p in np.asarray(pts) * SS], fill=a, width=max(1, round(w * SS)), joint='curve')

    def disc(p, r, a=255):
        x, y = np.asarray(p) * SS
        dr.ellipse([x - r * SS, y - r * SS, x + r * SS, y + r * SS], fill=a)

    def ring(p, r, w, a=255):
        x, y = np.asarray(p) * SS
        dr.ellipse([x - r * SS, y - r * SS, x + r * SS, y + r * SS], outline=a, width=max(1, round(w * SS)))

    def wline(pts, w, amp=.5, wl=40, a=255):
        line(_wobble(pts, amp, wl, rng), w, a)

    def inside(seg, m=25):  # trim a stroke where it would leave the canvas
        ok = (seg[:, 0] > m) & (seg[:, 0] < S - m) & (seg[:, 1] > m) & (seg[:, 1] < S - m)
        return seg if ok.all() else seg[:max(np.argmin(ok), 2)]

    def rays(base, d, lens, gap=0, sweep=None, w=1.05):
        """Fin: thin wobbly rays + wavy membrane edge; returns the membrane polygon."""
        q = np.linspace(0, 1, 10)[:, None]
        rs = [inside(b + (gap + q * (ln - gap)) * dd + (0 if sweep is None else .12 * ln * q ** 2 * sweep))
              for b, dd, ln in zip(base, d, lens)]
        tips = np.array([r[-1] for r in rs])
        for b, r in zip(base, rs):
            wline(r, w, .5, 35)
            if gap:  # faint root: converging roots must never darken into something that reads bold
                line([b, r[0]], .7, 110)
        margin = _wobble(_smooth(tips, base, .1), 2.2, 22, rng)
        line(margin, THIN)
        for b0, b1, t0, t1 in zip(base[:-1], base[1:], tips[:-1], tips[1:]):
            p0, p1 = (b0 + b1) / 2, (t0 + t1) / 2
            line([p0 + .3 * (p1 - p0), p0 + rng.uniform(.5, .75) * (p1 - p0)], .5, 200)
        return np.vstack([base, margin[::-1]])

    fins = []
    for kd, word in zip(kois, words):
        xy, tan, nrm = kd['xy'], kd['tan'], kd['nrm']
        fly = rng.uniform() < .35  # butterfly koi: long, flowing fins
        # scales + coding hatch groups (same drawing as fish.py)
        arcs = np.radians(np.linspace(-90, 90, 16))
        hatch = np.radians(np.linspace(40, 80, 6))
        allbits = np.concatenate([word, rng.integers(0, 2, len(CELLS) - len(word) + len(EXTRA))])
        for (u, v, du, dv, _), bit in zip(np.vstack([CELLS, EXTRA]), allbits):
            u0 = u - .55 * du
            line(xy(u0 + du * np.cos(arcs), v + .5 * dv * np.sin(arcs)), 1.5)
            side = -1 if bit else 1
            for rr in (.45, .6, .75):
                line(xy(u0 + rr * du * np.cos(hatch), v + side * rr * .5 * dv * np.sin(hatch)), 1.2)
        # bold outline, closed across the tail base: the decoder's silhouette
        line(_wobble(kd['poly'], .5, 70, rng), BOLD)
        line(_wobble(xy(np.ones(20), np.linspace(-1, 1, 20)), .3, 40, rng), BOLD * .85)
        # head: eyes on both sides, nostrils, gill arcs, barbels
        for sd in (-1, 1):
            e = xy(.085, .7 * sd)
            ring(e, 7.5, BOLD * .7)
            disc(e, 3.5)
            disc(e + 1.2 * tan(.085), 1.1, 0)
            ring(xy(.035, .35 * sd), 2.6, .9)
            vv = np.linspace(.45, 1, 12) * sd
            wline(xy(.2 - .04 * ((1 - np.abs(vv)) / .55) ** 2, vv), THIN, .4, 30)
            for (bu, bv, ln) in ((.012, .25, rng.uniform(26, 40)), (.045, .45, rng.uniform(18, 28))):
                b0 = xy(bu, bv * sd)
                dd = -tan(bu) * np.cos(.9) + nrm(bu) * sd * np.sin(.9)
                q = np.linspace(0, 1, 10)[:, None]
                line(b0 + q * ln * dd + 6 * q ** 2 * nrm(bu) * sd, 1.0)
            for _ in range(4):  # a few head dashes
                p = xy(rng.uniform(.04, .17), rng.uniform(.1, .55) * sd)
                line([p, p + rng.uniform(4, 8) * tan(.1)], .9, 220)
            # pectoral (big, flowing) and pelvic fins, on the outside of the outline
            pec, pel = ((150, 210), (80, 120)) if fly else ((85, 135), (40, 70))
            for (ua, ub, n, (lmin, lmax), tilt) in ((.17, .26, 11 if fly else 9, pec, .75), (.45, .51, 6, pel, .9)):
                us_ = np.linspace(ua, ub, n)
                out = nrm(us_) * sd
                tl = tilt * np.linspace(.75, 1.15, n)[:, None]
                d = np.cos(tl) * out + np.sin(tl) * tan(us_)
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                lens = rng.uniform(lmin, lmax) * (.7 + .3 * np.sin(np.pi * np.linspace(.15, 1, n)))
                fins.append(rays(xy(us_, sd), d, lens, gap=6, sweep=tan(us_[-1])))
        # tail: broad, softly forked, flowing; rays start off the bold base so they don't blur into it
        vs = np.linspace(-1, 1, rng.integers(13, 19))
        spread, fork = rng.uniform(.45, .7), rng.uniform(.1, .4)
        tl = rng.uniform(165, 195) if fly else rng.uniform(120, 170)
        th = spread * vs[:, None]
        d = np.cos(th) * tan(1.) + np.sin(th) * nrm(1.)
        fins.append(rays(xy(np.ones_like(vs), vs), d, tl * (1 - fork * (1 - np.abs(vs) ** 1.2)), gap=14))

    # --- washes (1x): water, koi bodies, patches, translucent fins ------------------------
    img = np.empty((S, S, 3), np.float32)
    img[:] = PAPER
    low = lambda n, sig: cv2.resize(rng.normal(0, 1, (n, n)).astype(np.float32), (S, S),
                                    interpolation=cv2.INTER_CUBIC) * sig
    water = _hsv(rng.uniform(.42, .52), rng.uniform(.12, .2), .94) / 255
    img *= 1 - (.8 + low(12, .08))[..., None] * (1 - water)
    bodies = np.zeros((S, S), np.uint8)
    for kd in kois:
        cv2.fillPoly(bodies, [kd['poly'].astype(np.int32)], 1)
    img = img * (1 - bodies[..., None]) + PAPER * bodies[..., None]
    # rounded-body shading: flanks a few % darker over a wide, smooth falloff, far gentler than the
    # decoder's background window, so it never tips a scale's up/down read
    edge = cv2.distanceTransform(bodies, cv2.DIST_L2, 3)
    img *= 1 - (.07 * np.exp(-edge / 45) * bodies)[..., None]
    # koi varieties: pale, high-value patches that follow the body (the decoder never sees them)
    layers = {k: np.zeros((S, S), np.float32) for k in ('red', 'gold', 'yellow', 'blue', 'cha')}
    t = np.linspace(0, 2 * np.pi, 90)

    def patch(layer, xy, uc, vc, ru, rv, val=1.):
        wob = 1 + .1 * np.sin(3 * t + rng.uniform(0, 6)) + .06 * np.sin(7 * t + rng.uniform(0, 6))
        cv2.fillPoly(layers[layer], [xy(uc + ru * wob * np.cos(t), vc + rv * wob * np.sin(t)).astype(np.int32)], val)

    for kd in kois:
        xy = kd['xy']
        kind = rng.choice(['kohaku', 'kohaku', 'tancho', 'maruten', 'ogon', 'yamabuki', 'asagi', 'chagoi', 'shiro'])
        if kind in ('ogon', 'yamabuki', 'chagoi'):
            cv2.fillPoly(layers[{'ogon': 'gold', 'yamabuki': 'yellow', 'chagoi': 'cha'}[kind]],
                         [kd['poly'].astype(np.int32)], .85)
        if kind == 'kohaku':  # stepped red blocks across the back, head to tail
            u = rng.uniform(.04, .12)
            while u < .85:
                ru = rng.uniform(.05, .11)
                patch('red', xy, u + ru, rng.uniform(-.25, .25), ru, rng.uniform(.8, 1.4))
                u += 2 * ru + rng.uniform(.04, .12)
        if kind in ('tancho', 'maruten'):  # the red crown
            patch('red', xy, .09, 0, .045, .5)
        if kind == 'maruten':
            for _ in range(rng.integers(1, 3)):
                patch('red', xy, rng.uniform(.35, .8), rng.uniform(-.2, .2), rng.uniform(.05, .1), rng.uniform(.7, 1.2))
        if kind == 'asagi':  # blue-grey back, red flanks and cheeks
            patch('blue', xy, .55, 0, .45, .55)
            for sd in (-1, 1):
                patch('red', xy, .5, 1.05 * sd, .4, .3)
    keep = cv2.erode(bodies, np.ones((5, 5), np.uint8)).astype(np.float32)
    for k, col in (('red', _hsv(rng.uniform(.01, .05), rng.uniform(.6, .8), .96)),
                   ('gold', _hsv(rng.uniform(.1, .13), rng.uniform(.45, .6), .97)),
                   ('yellow', _hsv(rng.uniform(.13, .16), rng.uniform(.35, .5), .98)),
                   ('blue', _hsv(rng.uniform(.55, .6), rng.uniform(.15, .25), .92)),
                   ('cha', _hsv(rng.uniform(.06, .09), rng.uniform(.3, .4), .9))):
        a = cv2.GaussianBlur(layers[k], (0, 0), 2.5) * keep
        img *= 1 - a[..., None] * (1 - col / 255)
    finm = np.zeros((S, S), np.float32)
    for poly in fins:
        cv2.fillPoly(finm, [poly.astype(np.int32)], .55)
    img += (PAPER - img) * cv2.GaussianBlur(finm, (0, 0), 3)[..., None] * (1 - bodies[..., None])

    # --- plants, placed in the water the koi leave free ---------------------------------
    q = 4
    fp = np.zeros((S // q, S // q), np.uint8)
    for poly in [kd['poly'] for kd in kois] + fins:
        cv2.fillPoly(fp, [(poly / q).astype(np.int32)], 1)
    fp = cv2.dilate(fp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 30 // q + 1,) * 2))

    def spot(r, tries=300):
        """A centre at least r px clear of koi, fins, other plants and the edge, or None."""
        free = cv2.distanceTransform(1 - fp, cv2.DIST_L2, 3) * q
        for _ in range(tries):
            p = rng.uniform(r + 30, S - r - 30, 2)
            if free[int(p[1]) // q, int(p[0]) // q] > r:
                cv2.circle(fp, (int(p[0]) // q, int(p[1]) // q), int(r / q) + 2, 1, -1)
                return p
        return None

    green, pink, leaf, stone = (np.zeros((S, S), np.float32) for _ in range(4))
    free = lambda: cv2.distanceTransform(1 - fp, cv2.DIST_L2, 3) * q
    clear = lambda pts, fr: (fr[np.clip(pts[:, 1] // q, 0, S // q - 1).astype(int),
                                np.clip(pts[:, 0] // q, 0, S // q - 1).astype(int)] > 0).all()

    for c_ in ANCHORS:  # anchors first: stones and reeds keep clear of them
        cv2.circle(stone, tuple(int(v) for v in c_), ANCHOR_R + 8, 1, -1)
        ring(c_, ANCHOR_R, ANCHOR_W)
        disc(c_, ANCHOR_DOT)
        cv2.circle(fp, tuple(int(v) // q for v in c_), (ANCHOR_R + 24) // q, 1, -1)

    # stone rim: rounded stones along the pond edge, wherever no koi or fin reaches
    per = np.concatenate([np.stack([np.linspace(0, S, 60), np.zeros(60)], -1), np.stack([np.full(60, S), np.linspace(0, S, 60)], -1),
                          np.stack([np.linspace(S, 0, 60), np.full(60, S)], -1), np.stack([np.zeros(60), np.linspace(S, 0, 60)], -1)])
    rimpath = per + (np.full(2, S / 2) - per) / np.linalg.norm(np.full(2, S / 2) - per, axis=1, keepdims=True) * 80
    fr = free()
    walk, k = 0., 0
    lens = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(rimpath, axis=0), axis=1))])
    while walk < lens[-1]:
        r = rng.uniform(32, 62)
        c_ = np.array([np.interp(walk, lens, rimpath[:, i]) for i in (0, 1)]) + rng.normal(0, 10, 2)
        walk += 2 * r * rng.uniform(.8, 1.0)
        tt = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        rad = r * (1 + .12 * np.sin(2 * tt + rng.uniform(0, 6)) + .07 * np.sin(5 * tt + rng.uniform(0, 6)))
        poly = c_ + rad[:, None] * np.stack([np.cos(tt), np.sin(tt)], -1)
        if poly.min() < 12 or poly.max() > S - 12 or not clear(poly, free()):  # off the edge; never overlapping
            continue
        cv2.fillPoly(stone, [poly.astype(np.int32)], rng.uniform(.7, 1))
        cv2.fillPoly(fp, [(poly / q).astype(np.int32)], 1)
        wline(np.vstack([poly, poly[:1]]), THIN, .6, 30)
        hl = np.linspace(3.6, 4.6, 10) + rng.uniform(-.3, .3)  # a worn highlight edge on the lit side
        line(c_ + .72 * r * np.stack([np.cos(hl), np.sin(hl)], -1), .6, 120)
        for _ in range(rng.integers(0, 3)):  # cracks and pits
            p0 = c_ + rng.normal(0, r * .3, 2)
            line(p0 + np.cumsum(rng.normal(0, 4, (5, 2)), 0), .7, 150)
        for _ in range(rng.integers(2, 6)):  # moss tufts where stone meets water
            p0 = poly[rng.integers(len(poly))]
            for _ in range(3):
                line([p0, p0 + rng.normal(0, 5, 2)], .6, 170)

    # reed clumps in the corners: tapered blades with a midrib
    for corner in ((0, 0), (S, 0), (0, S), (S, S)):
        c_ = np.array(corner, float)
        aim = np.arctan2(*(ctr - c_)[::-1])
        fr = free()
        for _ in range(rng.integers(2, 4)):
            root = c_ + rng.uniform(60, 200) * np.array([np.cos(aim + rng.uniform(-.6, .6)), np.sin(aim + rng.uniform(-.6, .6))])
            for _ in range(rng.integers(6, 12)):
                ang, ln, wd = aim + rng.uniform(-.7, .7), rng.uniform(160, 380), rng.uniform(5, 10)
                tt = np.linspace(0, 1, 24)[:, None]
                dvec, nvec = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
                mid = root + rng.normal(0, 8, 2) + ln * tt * dvec + rng.uniform(-.5, .5) * ln * tt ** 2 * nvec
                tg = np.gradient(mid, axis=0)
                nn = np.stack([-tg[:, 1], tg[:, 0]], -1) / np.linalg.norm(tg, axis=1, keepdims=True)
                half_w = wd * np.sin(np.pi * np.clip(tt[:, 0] * .95 + .05, 0, 1)) ** .6
                bl = np.vstack([mid + nn * half_w[:, None], (mid - nn * half_w[:, None])[::-1]])
                if not clear(bl, fr):
                    continue
                cv2.fillPoly(leaf, [bl.astype(np.int32)], rng.uniform(.6, 1))
                cv2.fillPoly(fp, [(bl / q).astype(np.int32)], 1)  # pads keep clear of the reeds
                line(np.vstack([bl, bl[:1]]), .9)
                line(mid[2:-3], .5, 150)

    # lily pads (more of them, some with lotus), plus loose ripples
    for _ in range(rng.integers(8, 14)):
        r = rng.uniform(55, 135)
        p = spot(r + 15)
        if p is None:
            continue
        a0, gapw = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(18, 30))
        t = np.linspace(a0 + gapw / 2, a0 + 2 * np.pi - gapw / 2, 90)
        rim = p + r * np.stack([np.cos(t), np.sin(t)], -1)
        cv2.fillPoly(green, [np.vstack([p, rim]).astype(np.int32)], 1)
        wline(np.vstack([p, rim, p]), THIN, .8, 45)
        for tv in np.linspace(t[0] + .25, t[-1] - .25, rng.integers(8, 14)):
            dd = np.array([np.cos(tv), np.sin(tv)])
            qq = np.linspace(.12, .9, 8)[:, None]
            line(p + qq * r * dd + 4 * qq ** 2 * np.array([-dd[1], dd[0]]), .8, 200)
        for f in sorted(rng.uniform(1.12, 1.45, rng.integers(1, 3))):  # ripple rings
            ta = rng.uniform(0, 2 * np.pi)
            tt = np.linspace(ta, ta + rng.uniform(2.5, 5), 80)
            line(p + f * r * np.stack([np.cos(tt), np.sin(tt)], -1), .9, 120)
        if rng.uniform() < .45:  # a lotus on the pad: three rings of petals, painted over the pad
            big = min(r * .75, rng.uniform(55, 75))
            for k_, (n, rr) in enumerate(((9, big), (7, big * .68), (5, big * .42))):
                for i in range(n):
                    ang = 2 * np.pi * (i + .5 * k_) / n + rng.uniform(-.1, .1)
                    dd, pp = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
                    tt = np.linspace(0, 1, 14)[:, None]
                    side = np.sin(np.pi * tt) ** .8 * rr * .34
                    petal = np.vstack([p + tt * rr * dd + side * pp, (p + tt * rr * dd - side * pp)[::-1]])
                    cv2.fillPoly(pink, [petal.astype(np.int32)], .6 + .2 * k_)
                    line(petal, 1.0)
                    line(p + tt[2:-3] * rr * dd, .45, 130)  # petal vein
            disc(p, 6, 0)
            ring(p, 6, .9)
            for i in range(8):
                disc(p + rng.normal(0, 2.5, 2), 1.0)
    for _ in range(rng.integers(3, 7)):
        p = spot(70)
        if p is not None:
            for f in np.linspace(1, rng.uniform(2, 3.2), rng.integers(2, 4)):
                ta = rng.uniform(0, 2 * np.pi)
                tt = np.linspace(ta, ta + rng.uniform(3, 6), 90)
                line(p + 20 * f * np.stack([np.cos(tt), np.sin(tt) * .8], -1), .9, 110)

    green *= 1 - np.clip(pink, 0, 1)  # petals sit on the pad, not tinted by it
    for layer, col, sig in ((stone, _hsv(rng.uniform(.06, .1), rng.uniform(.06, .12), .9), 1.5),
                            (leaf, _hsv(rng.uniform(.2, .26), rng.uniform(.3, .42), .86), 1.2),
                            (green, _hsv(rng.uniform(.24, .32), rng.uniform(.25, .38), .88), 2),
                            (pink, _hsv(rng.uniform(.93, .99), rng.uniform(.38, .5), .98), 1.5)):
        a = cv2.GaussianBlur(layer, (0, 0), sig) * .92
        img *= 1 - a[..., None] * (1 - col / 255)

    a = np.asarray(A.resize((S, S), Image.LANCZOS), np.float32)[..., None] / 255
    return np.clip(img * (1 - a) + INK * a, 0, 255).astype(np.uint8)


# --- decoder -----------------------------------------------------------------------------

def _ink(img, k=None):
    """Darkness against the local background (a k x k closing; default: a little wider than any stroke)."""
    v = img.max(2).astype(np.float32) + 1  # value channel: pale colour washes all but vanish
    k = k or _odd(max(7, min(v.shape) // 80))
    return np.clip(1 - v / cv2.morphologyEx(v, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)), 0, 1)


def _bold(ink):
    """Bold strokes only: thin strokes erased, then a relative threshold and a small closing. -> (bold, k)."""
    h, w = ink.shape
    # erase strokes clearly thinner than the bold outline (any number of them) before judging boldness;
    # sized from the drawing itself (99th-percentile stroke half-width), not the canvas
    dt = cv2.distanceTransform((ink > .5).astype(np.uint8), cv2.DIST_L2, 3)
    ko = min(8, int(1.6 * np.percentile(dt[dt > 0], 99))) if (dt > 0).any() else 0  # cap: smeared borders
    if ko >= 2:
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko,) * 2))
    b = cv2.GaussianBlur(ink, (0, 0), max(1., min(h, w) / 1500))
    bold = (b > .5 * np.percentile(b, 99.7)).astype(np.uint8)
    k = _odd(max(3, min(h, w) // 600))
    return cv2.morphologyEx(bold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), k


def _koi_masks(ink):
    """Bold strokes -> filled silhouettes, one per koi."""
    h, w = ink.shape
    bold, k = _bold(ink)
    _, lab = cv2.connectedComponents(1 - bold, connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    raw = (bold | ~np.isin(lab, border)).astype(np.uint8)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)
    m = cv2.morphologyEx(raw, cv2.MORPH_OPEN, disk)  # opening only decides what's a koi ...
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n < 2:
        return []
    # a koi never touches the frame; anything that does is border smear or clutter
    inner = (st[1:, 0] > 0) & (st[1:, 1] > 0) & (st[1:, 0] + st[1:, 2] < w) & (st[1:, 1] + st[1:, 3] < h)
    areas = st[1:, cv2.CC_STAT_AREA] * inner
    if areas.max() == 0:
        return []
    # ... the shape comes from the unopened mask just around it (2px): snout tip back, but nothing
    # that merely touches the body (JPEG ringing can make tail-ray roots read bold)
    return [raw & cv2.dilate((lab == i + 1).astype(np.uint8), np.ones((5, 5), np.uint8))
            for i in np.nonzero(areas > max(.35 * areas.max(), 1e-3 * h * w))[0]]


RING = (845, 935)  # spine radius of every koi (R0 870..910, +-25 per koi), canvas px


def _flanks(P, lo, hi, g, nth):
    """Raw flanks (first / last bold row per angle) -> smooth ones. Fin roots and pads push a flank outward, outline
    gaps let it fall in onto the scales, each over a narrow span of angle; a koi's real flank is smooth. A broad
    median drops both, a running mean smooths what's left (picked on real phone photos of a screen)."""
    A = lambda x: int(x * nth / (2 * np.pi)) | 1
    return [uniform_filter1d(median_filter(f, A(.25), mode='wrap'), A(.1), mode='wrap') for f in (lo, hi)]


def _pond_masks(ink):
    """Unwarped pond: closed outlines where they hold (exact), ring silhouettes for the koi whose outline leaked.
    Only the disc the koi swim in counts: whatever got warped in past the pond's edge (a dark bezel, window chrome)
    would otherwise set the bold threshold."""
    n = ink.shape[0]
    yy, xx = np.mgrid[:n, :n] - n / 2
    ink = ink * (np.hypot(xx, yy) < 1120 * n / S)
    masks = _koi_masks(ink)
    return masks + [m for m in _ring_masks(ink) if not any((m & k).sum() > .3 * m.sum() for k in masks)]


def _ring_masks(ink):
    """Silhouettes on an unwarped pond (RECT px), from the layout instead of closed outlines: every koi swims the
    ring, so unroll the band around it and, per angle, take the innermost and outermost bold pixel as the two flanks.
    A koi is a long run of angles with a body-wide gap between them; a broken outline only costs a few columns."""
    n = ink.shape[0]
    g, c = n / S, n / 2
    bold = _bold(ink)[0].astype(np.float32)
    r = np.arange(int((RING[0] - 12 - HW_RNG[1] - 10) * g), int((RING[1] + 12 + HW_RNG[1] + 10) * g))
    th = np.linspace(0, 2 * np.pi, int(2 * np.pi * np.mean(RING) * g), endpoint=False)
    P = cv2.remap(bold, (c + r[:, None] * np.cos(th)).astype(np.float32), (c + r[:, None] * np.sin(th)).astype(np.float32),
                  cv2.INTER_LINEAR) > .5
    seen = P.any(0)
    lo = np.where(seen, P.argmax(0), 0).astype(float)
    hi = np.where(seen, len(r) - 1 - P[::-1].argmax(0), 0).astype(float)
    lo, hi = _flanks(P, lo, hi, g, len(th))
    mid, wid = r[0] + (lo + hi) / 2, hi - lo
    body = (wid > 2 * 35 * g) & (wid < 2 * (HW_RNG[1] + 15) * g) & (mid > (RING[0] - 25) * g) & (mid < (RING[1] + 25) * g)
    w = int(.1 * len(th) / (2 * np.pi)) | 1  # bridge short breaks (a fin or pad bulging a flank); too-long runs get split below
    body = minimum_filter(maximum_filter(body.astype(np.uint8), w, mode='wrap'), w, mode='wrap') > 0
    if body.all() or not body.any():
        return []
    start = np.argmax(~body)  # unroll from a gap, so no koi straddles the seam
    idx = np.roll(np.arange(len(th)), -start)
    out, j = [], 0
    while j < len(idx):
        if not body[idx[j]]:
            j += 1
            continue
        k = j
        while k < len(idx) and body[idx[k]]:
            k += 1
        runs, done = [(j, k - 1, True, True)], []  # (first, last, extend at first, extend at last)
        while runs:  # koi swimming nose to tail can merge into one run: split it where the body is narrowest
            a, b, ea, eb = runs.pop()
            m0 = int(.3 * len(th) / (2 * np.pi))
            if (b - a + 1) * 2 * np.pi / len(th) > 1.1 and b - a > 2 * m0:
                cut = a + m0 + int(np.argmin(wid[idx[a + m0:b - m0]]))
                runs += [(a, cut - 1, ea, False), (cut + 1, b, False, eb)]
            else:
                done.append((a, b, ea, eb))
        for a, b, ea, eb in done:
            for step, ext in ((-1, ea), (1, eb)):  # out to the snout tip / tail base: narrower, same line
                e0 = a if step < 0 else b
                e1 = e0
                for _ in range(int(.15 * len(th) / (2 * np.pi)) if ext else 0):
                    q = idx[(e1 + step) % len(idx)]
                    if not (seen[q] and 6 * g < wid[q] and abs(mid[q] - mid[idx[e1 % len(idx)]]) < 30 * g):
                        break
                    e1 += step
                a, b = (e1, b) if step < 0 else (a, e1)
            if .45 < (b - a + 1) * 2 * np.pi / len(th) < 1.1:
                cols = idx[np.arange(a, b + 1) % len(idx)]
                # a fin root bulges a flank over a narrow span of angle; snout and tail taper over a long one:
                # 1-D grey closing of the inner flank / opening of the outer flank along the angle removes only bulges
                L = int(.08 * len(th) / (2 * np.pi)) | 1
                rl = r[0] + minimum_filter(maximum_filter(lo[cols], L, mode='nearest'), L, mode='nearest')
                rh = r[0] + maximum_filter(minimum_filter(hi[cols], L, mode='nearest'), L, mode='nearest')
                ang = th[cols]
                poly = np.vstack([np.stack([c + rl * np.cos(ang), c + rl * np.sin(ang)], -1),
                                  np.stack([c + rh * np.cos(ang), c + rh * np.sin(ang)], -1)[::-1]])
                m = np.zeros(ink.shape, np.uint8)
                cv2.fillPoly(m, [poly.round().astype(np.int32)], 1)
                out.append(m)
        j = k
    return out


def _resample_n(pts, n):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, i]) for i in (0, 1)], -1)


def _spine(m):
    """Silhouette -> 201-point centreline from one end to the other (direction not yet known)."""
    ys, xs = np.nonzero(m)
    P = np.stack([xs, ys], 1).astype(np.float64)
    mu = P.mean(0)
    e1 = np.linalg.svd(P - mu, full_matrices=False)[2][0]
    e2 = np.array([-e1[1], e1[0]])
    s, t = (P - mu) @ e1, (P - mu) @ e2
    cs, mids = [], []
    edges = np.linspace(s.min(), s.max(), 41)
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (s >= a) & (s < b)
        if sel.sum() > 5:
            cs.append((a + b) / 2)
            mids.append((t[sel].min() + t[sel].max()) / 2)
    if len(cs) < 10:
        return None
    co = np.polyfit(cs[2:-2], mids[2:-2], 3)
    sg = np.linspace(s.min(), s.max(), 300)
    pts = mu + sg[:, None] * e1 + np.polyval(co, sg)[:, None] * e2
    h, w = m.shape

    def march(p, d):  # walk out of the body along the end tangent to find the true tip
        d = d / np.linalg.norm(d)
        for _ in range(600):
            x, y = int(round(p[0] + d[0])), int(round(p[1] + d[1]))
            if not (0 <= x < w and 0 <= y < h) or not m[y, x]:
                break
            p = p + d
        return p

    def extend(sp, a):  # re-find both tips by marching out along the local tangent near each end
        b = len(sp) - 1 - a
        return _resample_n(np.vstack([march(sp[a], sp[a] - sp[2 * a]), sp[a:b + 1], march(sp[b], sp[b] - sp[b - a])]), 201)

    # the cubic is only a first guess (slices across the PCA axis run off-centre on a bent body);
    # the encoder's spine is the midline measured along its own normals, so converge onto that
    sp, mf = extend(pts, 20), m.astype(np.float32)
    for _ in range(3):
        _, nrm, _, _, mid, _ = _frame(mf, sp)
        sp = extend(sp + nrm * gaussian_filter1d(mid, 3)[:, None], 14)
    return sp


def _frame(m, sp):
    """Curved body frame for a spine running nose -> tail: points, normals, length, mid/half(u)."""
    L = np.linalg.norm(np.diff(sp, axis=0), axis=1).sum()
    tg = np.gradient(sp, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], -1)
    ts = np.arange(-.3 * L, .3 * L)
    X = sp[:, 0, None] + nrm[:, 0, None] * ts
    Y = sp[:, 1, None] + nrm[:, 1, None] * ts
    prof = cv2.remap(m, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_NEAREST) > 0
    lo = np.where(prof.any(1), ts[prof.argmax(1)], 0)
    hi = np.where(prof.any(1), ts[len(ts) - 1 - prof[:, ::-1].argmax(1)], 0)
    lo, hi = median_filter(lo, 15, mode='nearest'), median_filter(hi, 15, mode='nearest')  # fin-root bumps
    return sp, nrm, L, np.linspace(0, 1, len(sp)), (lo + hi) / 2, (hi - lo) / 2


def _sample(ink, fr, cells, off, tmpl):
    """Mean ink along each template curve of each cell -> (..., n, curves)."""
    sp, nrm, L, us, mid, half = fr
    cu, cv, du, dv = (x[:, None, None] for x in cells[:, :4].T)
    a, sl, b, st = (o[..., None, None] for o in np.moveaxis(np.asarray(off, float), -1, 0))
    U, V = np.broadcast_arrays(cu + (a + sl * cv + tmpl[0]) * du, cv + (b + st * cv + tmpl[1]) * dv)
    T = np.interp(U, us, mid) + V * np.interp(U, us, half)
    X, Y = (np.interp(U, us, sp[:, i]) + T * np.interp(U, us, nrm[:, i]) for i in (0, 1))
    k = tmpl[0].shape[-1]
    out = cv2.remap(ink, X.reshape(-1, k).astype(np.float32), Y.reshape(-1, k).astype(np.float32), cv2.INTER_LINEAR)
    return out.reshape(U.shape).mean(-1)


def _read_koi(ink, m):
    """One silhouette -> (koi index, chunk bytes) or None."""
    sp = _spine(m)
    if sp is None:
        return None
    m = m.astype(np.float32)
    # head end: koi are broad-shouldered and taper hard to the tail stalk
    fr = max((_frame(m, sp), _frame(m, sp[::-1])), key=lambda f: f[5][10:41].mean() - f[5][160:191].mean())
    off = np.zeros((COLS, 4))
    for step in (1, .4):  # local tracking: shift, slant, stretch per column, coarse then fine
        off = np.array([o + _COMBOS[np.argmax(_sample(ink, fr, CELLS[a:b], o + step * _COMBOS[:, None], ARC)
                                              .mean((-1, -2)))] * step if b > a else o
                        for o, a, b in zip(off, _STARTS[:-1], _STARTS[1:])])
        off = median_filter(off, size=(3, 1), mode='nearest')
    r = _sample(ink, fr, CELLS, off[_COL], HATCH)
    canon = (r[:, 0] > r[:, 1]).astype(np.uint8)
    for bits in (canon, 1 - canon[MIRROR]):  # as drawn, or seen in a mirror
        try:
            msg = bytes(_RS.decode(np.packbits(bits[:FISH_BITS]).tobytes())[0])
        except ReedSolomonError:
            continue
        if msg[0] >> 4 == 0xA and (msg[0] & 15) < NF:
            return msg[0] & 15, np.frombuffer(msg[1:], np.uint8)
    return None


def _anchors(img):
    """Corner bullseyes in a photo -> their 4 centres (full-res pixels, in ANCHORS order up to rotation), or None."""
    f = min(1., 1000 / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    b = (_ink(small, _odd(min(small.shape[:2]) // 25)) > .3).astype(np.uint8)  # wider than the centre dot, or it reads hollow
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    found = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if min(w, h) < 10 or not .5 < w / h < 2 or a < .15 * w * h:
            continue
        ring = np.pad(lab[y:y + h, x:x + w] == i, 1).astype(np.uint8)
        outside = ring.copy()
        cv2.floodFill(outside, None, (0, 0), 1)
        hole = outside == 0
        filled = ring.astype(bool) | hole
        if not .25 < hole.sum() / filled.sum() < .85 or filled.sum() < .6 * np.pi / 4 * w * h:
            continue
        dot = hole & np.pad(b[y:y + h, x:x + w], 1).astype(bool)  # ink inside the hole: the centre dot
        if not .03 < dot.sum() / filled.sum() < .3:
            continue
        ys, xs = np.nonzero(filled)
        dy, dx = np.nonzero(dot)
        c, cd = np.array([xs.mean(), ys.mean()]), np.array([dx.mean(), dy.mean()])
        if np.linalg.norm(c - cd) > .2 * max(w, h):
            continue
        found.append((c + [x - 1, y - 1], np.sqrt(filled.sum())))
    if len(found) < 3:
        return None
    found = sorted(found, key=lambda t: -t[1])[:8]
    best, quad = 0, None
    from itertools import combinations
    for k in (4, 3):
        for combo in combinations(found, k):
            sz = [t[1] for t in combo]
            if max(sz) > 2.5 * min(sz):
                continue
            pts = np.float32([t[0] for t in combo])
            if k == 3:  # one corner lost: the right-angle vertex is opposite the longest side; complete the parallelogram
                d = [np.linalg.norm(pts[(j + 1) % 3] - pts[(j + 2) % 3]) for j in range(3)]
                j = int(np.argmax(d))
                pts = np.vstack([pts, pts[(j + 1) % 3] + pts[(j + 2) % 3] - pts[j]])
            hull = cv2.convexHull(pts)
            area = cv2.contourArea(hull)
            if len(hull) == 4 and _squareish(hull[:, 0]) and area > best:
                best, quad = area, pts
        if quad is not None:
            break
    if quad is None or best < .04 * small.shape[0] * small.shape[1]:
        return None
    m = quad.mean(0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - m[1], quad[:, 0] - m[0]))]  # TL, TR, BR, BL (image y down)
    return quad / f


def _squareish(q):
    """A photo of the pond's square, however tilted, keeps every corner within 50..130 degrees and opposite sides within
    a factor 1.8; a quad that picked up a lily pad's centre instead of a corner doesn't."""
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        u, v = a - b, c - b
        if not 50 < np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))) < 130:
            return False
    sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    return max(sides[0], sides[2]) < 1.8 * min(sides[0], sides[2]) and max(sides[1], sides[3]) < 1.8 * min(sides[1], sides[3])


def _rectify(img):
    """Unwarp the pond onto a RECT x RECT canvas via its corner anchors (anchors blanked); the photo as is without them."""
    quad = _anchors(img)
    if quad is None:
        return img
    H = cv2.getPerspectiveTransform(ANCHORS * (RECT / S), quad.astype(np.float32))
    out = cv2.warpPerspective(img, H, (RECT, RECT), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    c = int((ANCHOR_M + ANCHOR_R + 20) * RECT / S)  # koi never come near the corners
    water = np.median(out.reshape(-1, 3)[::97], 0)  # the pond's water colour
    for y0, x0 in ((0, 0), (0, RECT - c), (RECT - c, 0), (RECT - c, RECT - c)):
        out[y0:y0 + c, x0:x0 + c] = water
    return out


def decode(img):
    rect = _rectify(img)
    ink = _ink(rect)
    masks = _pond_masks(ink) if rect is not img else _koi_masks(ink)
    if not masks:
        return None
    ink = cv2.GaussianBlur(ink, (0, 0), max(.7, .05 * CELLS[0, 2] * np.mean(L_RNG) * np.sqrt(ink.size) / S))
    chunks = {}
    for m in masks:
        res = _read_koi(ink, m)
        if res is not None:
            chunks.setdefault(res[0], res[1])
    if not chunks:
        return None
    out, erased = np.zeros(NF * CHUNK, np.uint8), []
    for f in range(NF):
        if f in chunks:
            out[f::NF] = chunks[f]
        else:
            erased += list(range(f, NF * CHUNK, NF))
    return np.unpackbits(out), erased


if __name__ == '__main__':  # round-trip self-check on one clean pond (first style attempt only)
    import time
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, NF * CHUNK * 8).astype(np.uint8)
    t0 = time.time()
    img = _render(bits, (0, 0))
    t1 = time.time()
    res = decode(img)
    print(f'render {t1 - t0:.1f}s decode {time.time() - t1:.1f}s')
    assert res is not None, 'not detected'
    assert not res[1], f'lost koi chunks: erased bytes {res[1][:6]}...'
    assert (res[0] == bits).all(), f'{(res[0] != bits).sum()} bit errors'
    print('ok')
