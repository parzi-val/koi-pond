"""Engraved fish: the payload lives in the shading hatch of each body scale.

Every coded scale gets one short inner hatch stroke on either its dorsal or its ventral
side; that choice is the bit. Both variants carry the same amount of ink, so the
engraving's tone is untouched, and the decoder only compares local ink above vs below
each scale centre, a differential read that shrugs off lighting, JPEG and mild blur.
The watercolour wash doubles as the sync mark: from its outline the decoder recovers the
nose, the tail cut and the dorsal side, then rebuilds the scale grid in body coordinates
(u: nose 0 -> tail 1, v: back -1 -> belly +1).
"""
import colorsys
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter

W, H, SS = 1024, 640, 2
# body shape ranges (the row schedule below is sized for the smallest body these allow)
L_RNG, HT_RNG, HB_RATIO, P_RNG, C_RNG, D_RNG = (680, 730), (145, 205), (.62, .8), (22, 32), (.55, .75), (.6, .9)
COLS, U0, U1 = 26, .25, .9    # coded scale columns span body fraction U0..U1
VM = .88                      # scales cover v in [-VM, VM]
DV_PX = 15                    # minimum scale height in px: below this the two hatch positions blur together


def _profile(u, Hx, c, d, P):
    return Hx * np.sin(np.pi * np.clip(u, 0, 1) ** c) ** d + P * u ** 2


def _min_half(u):
    g = np.linspace(0, 1, 5)
    c, d = np.meshgrid(C_RNG[0] + g * np.ptp(C_RNG), D_RNG[0] + g * np.ptp(D_RNG))
    h = lambda Hx: min(_profile(u, Hx, ci, di, P_RNG[0]) for ci, di in zip(c.ravel(), d.ravel()))
    return (h(HT_RNG[0]) + h(HT_RNG[0] * HB_RATIO[0])) / 2


def _rows(cols):
    du = (U1 - U0) / COLS
    return [int(2 * VM * _min_half(U0 + (c + .5) * du) / DV_PX - .5) for c in cols]


ROWS = _rows(range(COLS))
PAYLOAD_BYTES = 24
PARITY_BYTES = sum(ROWS) // 8 - PAYLOAD_BYTES
CODE_BITS = sum(ROWS) // 8 * 8  # leftover scales get filler bits


def _cells(rows, c0=0):
    """(u, v, du, dv) of each scale in bit order: columns nose->tail, rows back->belly, odd columns staggered."""
    du = (U1 - U0) / COLS
    out = []
    for c, n in enumerate(rows, c0):
        dv = 2 * VM / (n + .5)
        out += [(U0 + (c + .5) * du, -VM + (j + .5 + .5 * (c % 2)) * dv, du, dv) for j in range(n)]
    return np.array(out)


CELLS = _cells(ROWS)
EXTRA = _cells(_rows(range(COLS, COLS + 3)), COLS)  # decorative scales toward the tail, not read


def _smooth(tips, base, dip):
    """Fin membrane margin: soft scallops between ray tips, dipping toward the base by `dip`."""
    out = [tips[0]]
    for t0, t1, b0, b1 in zip(tips[:-1], tips[1:], base[:-1], base[1:]):
        ctrl = (t0 + t1) / 2 * (1 - dip) + (b0 + b1) / 2 * dip
        q = np.linspace(0, 1, 6)[1:, None]
        out += list((1 - q) ** 2 * t0 + 2 * q * (1 - q) * ctrl + q ** 2 * t1)
    return np.array(out)


def _hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1, s, v)) * 255


def encode(bits, z):
    """Render, then self-verify; re-roll the style (never the bits) if the clean render misreads."""
    for attempt in range(8):
        img = _render(bits, (z, attempt))
        res = decode(img)
        if res is not None and (res[0] == bits).all():
            return img
    return img  # ponytail: gives up after 8 styles; shows up as a clean-decode failure in the harness


def _render(bits, z):
    rng = np.random.default_rng(z)
    # --- shape -------------------------------------------------------------------------
    L, tail, face = rng.uniform(*L_RNG), rng.uniform(165, 205), rng.choice([-1, 1])
    Ht = rng.uniform(*HT_RNG)
    Hb, P = Ht * rng.uniform(*HB_RATIO), rng.uniform(*P_RNG)  # back always taller than belly: decoder uses it
    # depth peaks forward of centre: the decoder finds the head by it
    (ct, dt), (cb, db) = rng.uniform([C_RNG[0], D_RNG[0]], [C_RNG[1], D_RNG[1]], (2, 2))
    room = H - 40 - (Ht + Hb)  # vertical space left for fins
    DH, VH = min(rng.uniform(60, 125), .55 * room), min(rng.uniform(40, 85), .38 * room)
    cy = H / 2 + (Ht + DH - Hb - VH) / 2
    nose_x = (W - L - tail) / 2 if face > 0 else W - (W - L - tail) / 2

    def ht(u): return _profile(u, Ht, ct, dt, P)
    def hb(u): return _profile(u, Hb, cb, db, P)

    def xy(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        a, b = ht(u), hb(u)
        return np.stack([nose_x + face * u * L, cy + (b - a) / 2 + v * (a + b) / 2], -1)

    # --- colour --------------------------------------------------------------------------
    hue = rng.uniform()
    paper = _hsv(rng.uniform(.08, .14), rng.uniform(.04, .1), .96)
    back = _hsv(hue, rng.uniform(.45, .65), rng.uniform(.6, .75))
    belly = _hsv(hue + rng.uniform(-.08, .08), rng.uniform(.25, .35), rng.uniform(.88, .95))
    ink = tuple(int(c) for c in _hsv(rng.uniform(0, .12), rng.uniform(.3, .6), rng.uniform(.12, .2)))

    # --- paper + watercolour wash (multiplicative, darker back, pigment pooling at edges) --
    S2 = (H * SS, W * SS)
    small = lambda sig: cv2.resize(rng.normal(0, 1, (H // 16, W // 16)).astype(np.float32), S2[::-1],
                                   interpolation=cv2.INTER_CUBIC) * sig
    img = paper + cv2.GaussianBlur(rng.normal(0, 3, S2).astype(np.float32), (0, 0), 1.2)[..., None]
    uu = np.linspace(0, 1, 400)
    mask = np.zeros(S2, np.uint8)
    cv2.fillPoly(mask, [(np.vstack([xy(uu, -1), xy(uu[::-1], 1)]) * SS).astype(np.int32)], 1)
    u_col = np.clip((np.arange(S2[1]) / SS - nose_x) * face / L, 0, 1)
    top, bot = cy - ht(u_col), cy + hb(u_col)
    t = np.clip((np.arange(S2[0])[:, None] / SS - top) / np.maximum(bot - top, 1), 0, 1)
    wash = (back + (belly - back) * t[..., None] ** .8) * (1 + small(.04))[..., None] / 255
    # species markings live in the wash only, vary along u (so they can't tilt a scale's dorsal/ventral
    # read) and get deeper *and* more saturated so the decoder's chroma mask still sees body
    kind = rng.choice(['plain', 'bars', 'saddles'])
    if kind != 'plain':
        if kind == 'bars':
            k, ph = rng.integers(4, 8), rng.uniform()
            dark = (.5 + .5 * np.cos(2 * np.pi * (k * u_col + ph))) ** 3
        else:
            cs = rng.uniform(.3, .9, rng.integers(3, 6))
            dark = np.exp(-((u_col[:, None] - cs) / .035) ** 2).sum(1)
        dark = np.clip(dark * rng.uniform(.5, .8), 0, 1)[None] * (1 - t) ** 1.5 * (u_col > 0)[None]
        deep = _hsv(hue + rng.uniform(-.04, .04), rng.uniform(.75, .9), rng.uniform(.38, .5)) / 255
        wash = wash + (deep - wash) * dark[..., None]
    alpha = mask * (.8 + .2 * np.exp(-cv2.distanceTransform(mask, cv2.DIST_L2, 3) / (5 * SS)))
    img = img * (1 - alpha[..., None] * (1 - wash))

    # --- fins & other outside features: planned first so a neutral grey membrane tint can go
    # under the ink. Nothing out here carries colour, so the decoder's chroma mask never sees it.
    tailv = np.array([face, 0.])
    fins, blobs = [], []

    def inside(seg):  # trim a filament where it would leave the canvas
        ok = (seg[:, 0] > 8) & (seg[:, 0] < W - 8) & (seg[:, 1] > 8) & (seg[:, 1] < H - 8)
        return seg if ok.all() else seg[:np.argmin(ok)]

    def plan(base, d, lens, curl, w, dip, spiny=False, fil=()):
        q = np.linspace(0, 1, 9)[:, None]
        rs = [b + q * ln * dd + curl * ln * q ** 2 * tailv for b, dd, ln in zip(base, d, lens)]
        tips = np.array([b + (.86 if spiny else 1) * (r[-1] - b) for b, r in zip(base, rs)])
        fl = []
        for k, ext in fil:  # trailing filament continuing ray k, sweeping back
            r = rs[k]
            dd = (r[-1] - r[-2]) / np.linalg.norm(r[-1] - r[-2])
            q2 = np.linspace(0, 1, 16)[:, None]
            fl.append(inside(r[-1] + q2 * ext * dd + .6 * ext * q2 ** 2 * tailv))
        fins.append(dict(base=base, rays=rs, tips=tips, dip=dip, w=w, fil=fl))

    def outward(us, v):
        tang = xy(us + 1e-3, v) - xy(us - 1e-3, v)
        nrm = np.stack([tang[:, 1], -tang[:, 0]], -1)
        nrm *= np.sign(nrm[:, 1:2]) * np.sign(v)  # up for dorsal, down for ventral
        return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)

    def fin(u_a, u_b, v, n, height, profile, tilt, w=1.3, dip=.1, spiny=False, fil=()):
        us = np.linspace(u_a, u_b, n)
        tl = tilt * np.linspace(.8, 1.2, n)  # rays lean further back toward the rear of the fin
        d = np.cos(tl)[:, None] * outward(us, v) + np.sin(tl)[:, None] * tailv
        plan(xy(us, v), d, height * profile(np.linspace(0, 1, n)), .1, w, dip, spiny, fil)

    g = rng.uniform(.5, 1.5)
    if rng.uniform() < .5:  # one long dorsal fin, maybe a streaming lead ray
        fil = [(0, rng.uniform(.6, 1.3) * DH)] if rng.uniform() < .4 else []
        fin(.34, .74, -1, rng.integers(14, 20), DH, lambda s: .3 + .7 * np.sin(np.pi * s ** g), rng.uniform(.4, .8),
            fil=fil)
    else:                   # spiny + soft dorsal; spines may trail filaments
        fil = [(k, rng.uniform(.4, 1.1) * DH) for k in range(rng.integers(1, 4))] if rng.uniform() < .5 else []
        fin(.32, .52, -1, rng.integers(8, 11), DH, lambda s: .35 + .65 * np.sin(np.pi * s ** .7), .5, 1.7, .22, True,
            fil=fil)
        fin(.56, .76, -1, rng.integers(10, 14), DH * .75, lambda s: .4 + .6 * np.sin(np.pi * s ** g), .7)
    fin(.6, .84, 1, rng.integers(12, 18), VH, lambda s: .35 + .65 * np.sin(np.pi * s ** .8), .6)  # anal
    fin(.29, .39, 1, 6, VH * 1.2, lambda s: .6 + .4 * np.sin(np.pi * s), .9,                      # pelvic
        fil=[(0, rng.uniform(.5, 1.2) * VH)] if rng.uniform() < .35 else [])

    # pectoral: fans back over the gill cover, stopping short of the first coded column (u < U0)
    n = 7
    ang = np.radians(np.linspace(78, 38, n))  # from horizontal-back (upper rays) to down-back
    lens = np.minimum(rng.uniform(60, 85) * (.75 + .25 * np.sin(np.pi * np.linspace(.2, 1, n))),
                      (U0 - .02 - .155) * L / np.sin(ang))
    plan(xy(np.linspace(.15, .16, n), np.linspace(.28, .5, n)), np.stack([face * np.sin(ang), np.cos(ang)], -1),
         lens, 0, 1.0, .1)

    # caudal fin: forked (maybe streaming lobes), emarginate, truncate or rounded
    fork, spread = rng.choice([.5, .25, .05, -.25]) + rng.uniform(-.05, .05), rng.uniform(.5, .8)
    vs = np.linspace(-1, 1, rng.integers(15, 21))
    th = spread * vs
    ext = min(rng.uniform(.25, .5) * tail, (W - L - tail) / 2 - 15)
    plan(xy(np.ones_like(vs), vs), np.stack([face * np.cos(th), np.sin(th)], -1),
         tail * (1 - fork * (1 - np.abs(vs) ** 1.2)), 0, 1.3, .06,
         fil=[(0, ext), (len(vs) - 1, ext)] if fork > .3 and rng.uniform() < .6 else [])

    extra = rng.choice(['none', 'adipose', 'finlets'])
    if extra == 'adipose':  # small fleshy fin between dorsal and tail
        us = np.linspace(.8, .88, 14)
        q = np.linspace(0, 1, 14)[:, None]
        lobe = xy(us, -1) + outward(us, -1) * rng.uniform(14, 22) * np.sin(np.pi * q) ** .6 + 6 * q ** 2 * tailv
        blobs.append(np.vstack([xy(us, -1), lobe[::-1]]))
    elif extra == 'finlets':  # tuna-style flags on both edges of the tail stalk
        for v in (-1, 1):
            for u_ in np.linspace(.86, .96, 5):
                b0, b1 = xy(u_, v), xy(u_ + .012, v)
                tip = b0 + outward(np.array([u_]), v)[0] * rng.uniform(10, 15) + 12 * tailv
                blobs.append(np.array([b0, tip, b1]))

    tint = np.zeros(S2, np.float32)
    for f in fins:  # darker near the base, fading toward the margin
        m = np.zeros(S2, np.uint8)
        poly = np.vstack([f['base'], _smooth(f['tips'], f['base'], f['dip'])[::-1]])
        cv2.fillPoly(m, [(poly * SS).astype(np.int32)], 1)
        bl = np.ones(S2, np.uint8)
        cv2.polylines(bl, [(f['base'] * SS).astype(np.int32)], False, 0, 3)
        span = max(np.linalg.norm(f['tips'] - f['base'], axis=1).mean(), 1)
        fade = np.exp(-cv2.distanceTransform(bl, cv2.DIST_L2, 3) / SS / (.5 * span))
        tint = np.maximum(tint, m * (.05 + .12 * fade))
    for poly in blobs:
        m = np.zeros(S2, np.uint8)
        cv2.fillPoly(m, [(poly * SS).astype(np.int32)], 1)
        tint = np.maximum(tint, m * .16)
    img = img * (1 - cv2.GaussianBlur(tint, (0, 0), SS)[..., None])

    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    def line(pts, w, fill=ink):
        draw.line([tuple(p) for p in np.asarray(pts) * SS], fill=fill, width=max(1, round(w * SS)), joint='curve')

    def disc(p, r, fill=ink):
        x, y = np.asarray(p) * SS
        draw.ellipse([x - r * SS, y - r * SS, x + r * SS, y + r * SS], fill=fill)

    # --- scales: imbricate half-ellipses (tips meet the previous column's apex) + coding hatch
    arc = np.radians(np.linspace(-90, 90, 16))
    hatch = np.radians(np.linspace(35, 85, 7))
    allbits = np.concatenate([bits, rng.integers(0, 2, len(CELLS) - len(bits) + len(EXTRA))])
    for (u, v, du, dv), bit in zip(np.vstack([CELLS, EXTRA]), allbits):
        u0 = u - .55 * du
        line(xy(u0 + du * np.cos(arc), v + .5 * dv * np.sin(arc)), 1.3)
        side = -1 if bit else 1  # bit 1: hatch on the dorsal side
        for rr in ([.6, .8] if v < -.35 else [.6]):  # back scales get a second hatch: darker tone
            line(xy(u0 + rr * du * np.cos(hatch), v + side * rr * .5 * dv * np.sin(hatch)), 1.6)

    def taper(pts, w0, w1, fill=ink):  # engraved stroke: swells at the root, thins to the tip
        pts = np.asarray(pts)
        for k in range(1, len(pts)):
            w = w0 + (w1 - w0) * k / (len(pts) - 1)
            line(pts[k - 1:k + 1], w, fill)
            disc(pts[k], w / 2, fill)

    # --- lateral line: pores on the scale apexes, well away from the hatch sample curves ---
    allc = np.vstack([CELLS, EXTRA])
    for c0 in np.unique(allc[:, 0]):
        col = allc[allc[:, 0] == c0]
        u, v, du, dv = col[np.argmin(np.abs(col[:, 1] - (-.4 + .12 * np.sin(np.pi * c0))))]
        disc(xy(u + .45 * du, v), 2.0)

    # --- outline -------------------------------------------------------------------------
    taper(xy(uu[::4], -1), 2.8, 1.5)
    taper(xy(uu[::4], 1), 2.4, 1.3)

    # --- fins (inked over their tint) ------------------------------------------------------
    for f in fins:
        for r in f['rays']:
            taper(r, f['w'], .45 * f['w'])
        line(_smooth(f['tips'], f['base'], f['dip']), .8)
        for b0, b1, t0, t1 in zip(f['base'][:-1], f['base'][1:], f['tips'][:-1], f['tips'][1:]):
            p0, p1 = (b0 + b1) / 2, (t0 + t1) / 2  # one faint striation per gap
            line([p0 + .25 * (p1 - p0), p0 + rng.uniform(.5, .7) * (p1 - p0)], .4)
        for seg in f['fil']:
            if len(seg) > 1:
                taper(seg, .5 * f['w'], .25)
    for poly in blobs:
        line(np.vstack([poly, poly[:1]]), .9)

    # --- head ------------------------------------------------------------------------------
    for k, uo in enumerate([.215, .18]):  # gill covers
        vv = np.linspace(-.85, .88, 30)
        taper(xy(uo + .045 * np.cos(vv * np.pi / 2), vv), 2.0 - .6 * k, 1.0 - .3 * k)
    for f in (.3, .55, .8):  # opercle shading, concentric with its edge
        vv = np.linspace(-.15, .75, 14)
        line(xy(.18 + .035 * f + .045 * np.cos(vv * np.pi / 2), vv), .55)
    for v in np.linspace(-.93, -.52, 6):  # contour hatching on the brow
        line(xy(np.linspace(.04 + .06 * (v + 1), .17, 12), v), .8)
    for v in np.linspace(.6, .9, 4):
        line(xy(np.linspace(.05, .15, 10), v), .6)
    for v in np.linspace(.1, .85, 9):  # serrated preopercle edge
        p = xy(.18 + .045 * np.cos(v * np.pi / 2), v)
        line([p, p + 4 * tailv + np.array([0, 1.5])], .7)
    if rng.uniform() < .4:  # barbels from the mouth corner, one pair (far one fainter)
        corner = xy(.07, .3)
        for k, (ln, w) in enumerate([(rng.uniform(55, 95), 1.5), (rng.uniform(40, 70), .9)]):
            q = np.linspace(0, 1, 20)
            a_ = np.radians(15 + 50 * q + 12 * np.sin(3 * np.pi * q) + 8 * k)
            steps = np.stack([face * np.sin(a_), np.cos(a_)], -1) * ln / 20
            taper(inside(corner + np.array([0, 3 * k]) + np.cumsum(steps, 0)), w, .3)
    eye_u, eye_v = .1, -.28
    re = .16 * (ht(eye_u) + hb(eye_u)) / 2
    for _ in range(70):  # cheek stipple, clear of the eye
        p = xy(rng.uniform(.06, .17), rng.uniform(.05, .75))
        if np.linalg.norm(p - xy(eye_u, eye_v)) > 1.6 * re:
            disc(p, rng.uniform(.4, .7))
    mu = np.linspace(.004, .07, 10)
    taper(xy(mu, .15 + 2 * (mu - .004)), 1.8, .8)  # mouth
    taper(xy(mu[2:], .3 + 2.2 * (mu[2:] - .01)), 1.0, .5)  # lower jaw
    nost = xy(.045, -.2)
    draw.ellipse([*(nost - 1.8) * SS, *(nost + 1.8) * SS], outline=ink, width=round(.8 * SS))
    eye = xy(eye_u, eye_v)
    disc(eye, re, (245, 240, 230))
    draw.ellipse([*(eye - re) * SS, *(eye + re) * SS], outline=ink, width=round(1.6 * SS))
    draw.ellipse([*(eye - .75 * re) * SS, *(eye + .75 * re) * SS], outline=ink, width=round(.6 * SS))
    disc(eye, .5 * re)
    disc(eye + np.array([-.2 * face, -.25]) * re, .15 * re, (250, 250, 245))
    lid = np.radians(np.linspace(200, 340, 12))
    line(eye + 1.3 * re * np.stack([np.cos(lid), np.sin(lid)], -1), .8)  # brow ridge over the eye
    return np.asarray(im.resize((W, H), Image.LANCZOS))


# --- decoder -----------------------------------------------------------------------------

def _odd(k):
    return int(k) | 1


def _body(img):
    """Watercolour wash -> (body mask, nose tip, tail-cut centre)."""
    h, w = img.shape[:2]
    f = img.astype(np.float32) + 1
    k = _odd(max(5, min(h, w) // 90))
    f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))  # erase ink lines
    small = cv2.resize(f, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    ks = _odd(.6 * min(small.shape[:2]))
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    paper = cv2.resize(cv2.GaussianBlur(paper, (0, 0), 2), (w, h), interpolation=cv2.INTER_LINEAR)
    r = f / paper  # paper -> (1,1,1): cancels lighting gradients and colour casts
    chroma = cv2.GaussianBlur(r.max(2) - r.min(2), (0, 0), 1.5)
    _, m = cv2.threshold(np.clip(chroma * 510, 0, 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n < 2:
        return None
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    _, lab = cv2.connectedComponents(1 - m, connectivity=4)  # fill holes (eye etc.)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    m[~np.isin(lab, border)] = 1
    if m.sum() < .01 * h * w:
        return None

    ys, xs = np.nonzero(m)
    P = np.stack([xs, ys], 1).astype(np.float64)
    mu = P.mean(0)
    e1 = np.linalg.svd(P - mu, full_matrices=False)[2][0]
    s, t = (P - mu) @ e1, (P - mu) @ [-e1[1], e1[0]]
    span = np.ptp(s)
    width = lambda a: np.ptp(t[np.abs(s - a) < .01 * span])
    fr = (.05, .1, .15, .2)
    if sum(width(s.min() + f * span) for f in fr) < sum(width(s.max() - f * span) for f in fr):  # head end is deeper
        e1 = -e1
    for _ in range(2):  # nose tip and tail-cut centre, pushed out to the extreme along the axis
        s = (P - mu) @ e1
        a, b = s < s.min() + .005 * span, s > s.max() - .01 * span
        nose = P[a].mean(0) + (s.min() - s[a].mean()) * e1
        ped = P[b].mean(0) + (s.max() - s[b].mean()) * e1
        e1 = (ped - nose) / np.linalg.norm(ped - nose)
    return m, nose, ped


def _frame(m, nose, ped):
    """Body frame for a given nose/tail-cut: axis e1, e2 toward the belly, length, profile mid/half(u)."""
    L = np.linalg.norm(ped - nose)
    e1 = (ped - nose) / L
    e2 = np.array([-e1[1], e1[0]])
    us, ts = np.linspace(0, 1, 201), np.arange(-.6 * L, .6 * L)
    X = nose[0] + us[:, None] * L * e1[0] + ts * e2[0]
    Y = nose[1] + us[:, None] * L * e1[1] + ts * e2[1]
    prof = cv2.remap(m, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_NEAREST) > 0
    top = np.where(prof.any(1), ts[prof.argmax(1)], 0)
    bot = np.where(prof.any(1), ts[len(ts) - 1 - prof[:, ::-1].argmax(1)], 0)
    if bot.sum() > -top.sum():  # back is the taller side; make e2 point to the belly
        e2, top, bot = -e2, -bot, -top
    return nose, e1, e2, L, us, (top + bot) / 2, (bot - top) / 2


def _ink(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) + 1
    k = _odd(max(7, min(g.shape) // 30))  # wider than a blurred line, narrower than the wash gradient
    return np.clip(1 - g / cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)), 0, 1)


# templates, in cell units relative to the scale centre: (du offsets, dv offsets), shape (curves, points)
_HT = np.radians(np.linspace(35, 85, 7))
HATCH = (np.broadcast_to(-.55 + .6 * np.cos(_HT), (2, 7)), np.array([[-1], [1]]) * .3 * np.sin(_HT))  # dorsal, ventral
_AT = np.radians(np.linspace(-80, 80, 9))
ARC = ((-.55 + np.cos(_AT))[None], (.5 * np.sin(_AT))[None])  # the scale outline: same in every scale, bit-free
# local grid tracking per column: shift along u (x du), slant (x du per unit v), shift along v (x dv),
# stretch along v (x dv per unit v)
_COMBOS = np.stack(np.meshgrid(*[np.linspace(-r, r, n) for r, n in [(.35, 7), (.3, 5), (.25, 5), (.3, 5)]],
                               indexing='ij'), -1).reshape(-1, 4)
_STARTS = np.cumsum([0] + ROWS)


def _sample(ink, frame, cells, off, tmpl):
    """Mean ink along each template curve of each cell -> (..., n, curves).
    off (..., n|1, 4) nudges cells by the local tracking offsets."""
    nose, e1, e2, L, us, mid, half = frame
    cu, cv, du, dv = (x[:, None, None] for x in cells.T)
    a, sl, b, st = (o[..., None, None] for o in np.moveaxis(np.asarray(off, float), -1, 0))
    U, V = np.broadcast_arrays(cu + (a + sl * cv + tmpl[0]) * du, cv + (b + st * cv + tmpl[1]) * dv)
    T = np.interp(U, us, mid) + V * np.interp(U, us, half)
    X, Y = (nose[i] + U * L * e1[i] + T * e2[i] for i in (0, 1))
    k = tmpl[0].shape[-1]
    out = cv2.remap(ink, X.reshape(-1, k).astype(np.float32), Y.reshape(-1, k).astype(np.float32), cv2.INTER_LINEAR)
    return out.reshape(U.shape).mean(-1)


def decode(img):
    b = _body(img)
    if b is None:
        return None
    m, nose, ped = b
    m = m.astype(np.float32)
    L = np.linalg.norm(ped - nose)
    ink = cv2.GaussianBlur(_ink(img), (0, 0), max(.7, .05 * CELLS[0, 2] * L))  # widen capture a little
    # alignment always maximises ink on the scale outlines (bit-free); the bits are read afterwards
    # 1) global: the blunt nose tip is poorly defined across the head; slide it along the perpendicular
    perp = np.array([-(ped - nose)[1], (ped - nose)[0]]) / L
    frame = max((_frame(m, nose + d * perp, ped) for d in np.linspace(-.03, .03, 13) * L),
                key=lambda f: _sample(ink, f, CELLS, np.zeros(4), ARC).mean())
    # 2) local: perspective / paper curl bend the columns; track each column's shift, slant and
    #    stretch, coarse then fine around the coarse answer
    off = np.zeros((COLS, 4))
    for step in (1, .4):
        off = np.array([o + _COMBOS[np.argmax(_sample(ink, frame, CELLS[a:b], o + step * _COMBOS[:, None], ARC)
                                              .mean((-1, -2)))] * step
                        for o, a, b in zip(off, _STARTS[:-1], _STARTS[1:])])
        off = median_filter(off, size=(3, 1), mode='nearest')  # one noisy column can't drag the grid
    r = _sample(ink, frame, CELLS, np.repeat(off, ROWS, 0), HATCH)
    return (r[:CODE_BITS, 0] > r[:CODE_BITS, 1]).astype(np.uint8), []


if __name__ == '__main__':  # round-trip self-check on clean renders (first style attempt only)
    rng = np.random.default_rng(1)
    for z in range(5):
        bits = rng.integers(0, 2, CODE_BITS).astype(np.uint8)
        res = decode(_render(bits, (z, 0)))
        assert res is not None, f'z={z}: not detected'
        assert (res[0] == bits).all(), f'z={z}: {(res[0] != bits).sum()} bit errors'
    print('ok')
