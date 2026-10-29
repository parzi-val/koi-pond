"""Fishdraw-style plate: the payload lives in the shading hatches of the body scales.

Every coded scale carries a small group of hatch strokes on either its dorsal or its ventral
side; that choice is the bit. Both variants carry the same ink, so the drawing's tone is
untouched, and the decoder only compares local ink above vs below each scale centre.

Sync comes from the line work itself: the body outline, tail base, gill line and eye are
the only bold strokes, so keeping the darkest blurred strokes and filling what they enclose
gives the silhouette without the (thin-lined) fins. From it: nose, tail base, back vs belly,
then the scale grid in body coordinates (u: nose 0 -> tail 1, v: back -1 -> belly +1).
Per-column tracking on the scale outlines (identical in every scale, bit-free) then absorbs
perspective and paper curl.
"""
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter

W, H, SS = 1024, 640, 2
PAPER, INK = np.array([255., 250., 240.]), (24, 20, 18)  # floral white, near-black
BOLD, THIN = 3.6, 1.4  # the decoder tells silhouette lines from everything else by weight
# body shape ranges (the row schedule below is sized for the smallest body these allow)
L_RNG, HT_RNG, HB_RATIO, P_RNG = (660, 720), (165, 210), (.62, .8), (22, 32)
C_RNG, D_RNG, K_RNG = (.55, .75), (.6, .9), (.03, .07)
COLS, U0, U1 = 26, .25, .9    # coded scale columns span body fraction U0..U1
VM = .88                      # scales cover v in [-VM, VM]
DV_PX = 15                    # minimum scale height in px: below this the two hatch positions blur together
PEC_U, PEC_V = .37, .05       # scales under the pectoral fin (u < PEC_U, v > PEC_V) are hidden, not coded


def _profile(u, Hx, c, d, P, k):
    """Half-depth of the body: deepest forward of centre, pointed snout (k), tail stalk P."""
    u = np.clip(u, 0, 1)
    return Hx * np.sin(np.pi * u ** c) ** d * (1 - np.exp(-u / k)) + P * u ** 2


def _min_half(u):
    g = np.linspace(0, 1, 5)
    c, d = np.meshgrid(C_RNG[0] + g * np.ptp(C_RNG), D_RNG[0] + g * np.ptp(D_RNG))
    h = lambda Hx: min(_profile(u, Hx, ci, di, P_RNG[0], K_RNG[1]) for ci, di in zip(c.ravel(), d.ravel()))
    return (h(HT_RNG[0]) + h(HT_RNG[0] * HB_RATIO[0])) / 2


def _cells(cols):
    """(u, v, du, dv, col) of each scale in bit order: columns nose->tail, rows back->belly, odd columns staggered."""
    du = (U1 - U0) / COLS
    out = []
    for c in cols:
        n = int(2 * VM * _min_half(U0 + (c + .5) * du) / DV_PX - .5)
        dv = 2 * VM / (n + .5)
        out += [(U0 + (c + .5) * du, -VM + (j + .5 + .5 * (c % 2)) * dv, du, dv, c) for j in range(n)]
    return np.array(out)


_ALL = _cells(range(COLS))
_HID = (_ALL[:, 0] < PEC_U) & (_ALL[:, 1] > PEC_V)
CELLS = _ALL[~_HID]
EXTRA = np.vstack([_ALL[_HID], _cells(range(COLS, COLS + 3))])  # decorative: under the pectoral, near the tail
PAYLOAD_BYTES = 24
PARITY_BYTES = len(CELLS) // 8 - PAYLOAD_BYTES
CODE_BITS = len(CELLS) // 8 * 8  # leftover scales get filler bits


def _resample(pts, step=2.):
    pts = np.asarray(pts, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    t = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
    return np.stack([np.interp(t, s, pts[:, i]) for i in (0, 1)], -1)


def _wobble(pts, amp, wl, rng):
    """Hand-drawn wobble: push a polyline along its normal by smooth noise (amp px, wavelength wl px)."""
    pts = _resample(pts)
    if len(pts) < 3 or amp == 0:
        return pts
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    tg = np.gradient(pts, axis=0)
    n = np.stack([-tg[:, 1], tg[:, 0]], -1) / (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9)
    a, b = rng.uniform(0, 2 * np.pi, 2)
    return pts + n * (amp * (.7 * np.sin(2 * np.pi * s / wl + a) + .3 * np.sin(2 * np.pi * s / (.37 * wl) + b)))[:, None]


def _smooth(tips, base, dip):
    """Fin membrane margin: soft scallops between ray tips, dipping toward the base by `dip`."""
    out = [tips[0]]
    for t0, t1, b0, b1 in zip(tips[:-1], tips[1:], base[:-1], base[1:]):
        ctrl = (t0 + t1) / 2 * (1 - dip) + (b0 + b1) / 2 * dip
        q = np.linspace(0, 1, 6)[1:, None]
        out += list((1 - q) ** 2 * t0 + 2 * q * (1 - q) * ctrl + q ** 2 * t1)
    return np.array(out)


_SYL = ['pa', 'che', 'ri', 'cto', 'lu', 'mae', 'thy', 'gno', 'sca', 'pel', 'ori', 'chro', 'bra', 'sti', 'dae',
        'myx', 'lo', 'pho', 'ce', 'nu', 'ty', 'gra', 'phi', 'ae']


def _name(rng):
    word = lambda n: ''.join(rng.choice(_SYL, n))
    return f"{word(rng.integers(2, 4)).capitalize()}{rng.choice(['us', 'a', 'is', 'ichthys'])} " \
           f"{word(rng.integers(1, 3))}{rng.choice(['is', 'us', 'ae', 'a', 'ensis'])}."


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
    L, tail, face = rng.uniform(*L_RNG), rng.uniform(150, 190), rng.choice([-1, 1])
    Ht = rng.uniform(*HT_RNG)
    Hb, P = Ht * rng.uniform(*HB_RATIO), rng.uniform(*P_RNG)  # back always taller than belly: decoder uses it
    # depth peaks forward of centre: the decoder finds the head by it
    (ct, dt), (cb, db) = rng.uniform([C_RNG[0], D_RNG[0]], [C_RNG[1], D_RNG[1]], (2, 2))
    k = rng.uniform(*K_RNG)
    room = H - 80 - (Ht + Hb)  # vertical space left for fins (and the caption)
    DH, VH = min(rng.uniform(60, 120), .55 * room), min(rng.uniform(45, 90), .42 * room)
    cy = (H - 34) / 2 + (Ht + DH - Hb - VH) / 2
    nose_x = (W - L - tail) / 2 if face > 0 else W - (W - L - tail) / 2

    def ht(u): return _profile(u, Ht, ct, dt, P, k)
    def hb(u): return _profile(u, Hb, cb, db, P, k)

    def xy(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        a, b = ht(u), hb(u)
        return np.stack([nose_x + face * u * L, cy + (b - a) / 2 + v * (a + b) / 2], -1)

    def uv(p):  # inverse of xy, for keeping features inside their allowed body region
        u = (p[..., 0] - nose_x) * face / L
        a, b = ht(u), hb(u)
        return u, (p[..., 1] - cy - (b - a) / 2) / ((a + b) / 2)

    tailv = np.array([face, 0.])
    S2 = (H * SS, W * SS)
    img = PAPER + cv2.GaussianBlur(rng.normal(0, 2, S2).astype(np.float32), (0, 0), 1.2)[..., None]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    def line(pts, w, fill=INK):
        draw.line([tuple(p) for p in np.asarray(pts) * SS], fill=fill, width=max(1, round(w * SS)), joint='curve')

    def disc(p, r, fill=INK):
        x, y = np.asarray(p) * SS
        draw.ellipse([x - r * SS, y - r * SS, x + r * SS, y + r * SS], fill=fill)

    def ring(p, r, w):
        x, y = np.asarray(p) * SS
        draw.ellipse([x - r * SS, y - r * SS, x + r * SS, y + r * SS], outline=INK, width=max(1, round(w * SS)))

    def wline(pts, w, amp=.5, wl=40):
        line(_wobble(pts, amp, wl, rng), w)

    def inside(seg):  # trim a stroke where it would leave the plate
        ok = (seg[:, 0] > 20) & (seg[:, 0] < W - 20) & (seg[:, 1] > 20) & (seg[:, 1] < H - 40)
        return seg if ok.all() else seg[:max(np.argmin(ok), 1)]

    # --- scales: imbricate half-ellipses + the coding hatch group --------------------------
    arc = np.radians(np.linspace(-90, 90, 16))
    hatch = np.radians(np.linspace(40, 80, 6))
    allbits = np.concatenate([bits, rng.integers(0, 2, len(CELLS) - len(bits) + len(EXTRA))])
    for (u, v, du, dv, _), bit in zip(np.vstack([CELLS, EXTRA]), allbits):
        u0 = u - .55 * du
        line(xy(u0 + du * np.cos(arc), v + .5 * dv * np.sin(arc)), 1.5)
        side = -1 if bit else 1  # bit 1: hatches on the dorsal side
        for rr in (.45, .6, .75):
            line(xy(u0 + rr * du * np.cos(hatch), v + side * rr * .5 * dv * np.sin(hatch)), 1.2)
    # lateral line: pores on the scale apexes, well away from the hatch sample curves
    allc = np.vstack([CELLS, EXTRA])
    for c0 in np.unique(allc[:, 0]):
        col = allc[allc[:, 0] == c0]
        u, v, du, dv, _ = col[np.argmin(np.abs(col[:, 1] - (-.4 + .12 * np.sin(np.pi * c0))))]
        disc(xy(u + .45 * du, v), 1.8)

    # --- pectoral: lies over the body, hiding the (uncoded) scales beneath -----------------
    n = 9
    pb = xy(np.linspace(.235, .245, n), np.linspace(.2, .45, n))
    ang = np.radians(np.linspace(84, 50, n) + rng.uniform(-6, 6))  # upper rays run back, lower ones back-down
    d = np.stack([face * np.sin(ang), np.cos(ang)], -1)
    pl = rng.uniform(95, 130) * (.7 + .3 * np.sin(np.pi * np.linspace(.25, 1, n)))
    for i in range(n):  # shorten rays until their tips sit inside the hidden region
        while True:
            tu, tv = uv(pb[i] + pl[i] * d[i])
            if tu < PEC_U - .018 and PEC_V + .1 < tv < .85 or pl[i] < 15:
                break
            pl[i] *= .96
    ptips = pb + pl[:, None] * d
    margin = _wobble(_smooth(ptips, pb, .08), 1.5, 18, rng)
    draw.polygon([tuple(p) for p in np.vstack([pb, margin[::-1]]) * SS], fill=tuple(PAPER.astype(int)))
    for b, t in zip(pb, ptips):
        wline(np.vstack([b, t]), 1.1, .4, 30)
    line(margin, THIN)

    # --- body outline (bold), closed at the tail base: this is the decoder's silhouette ----
    uu = np.linspace(0, 1, 300)
    line(_wobble(np.vstack([xy(uu[::-1], -1), xy(uu, 1)]), .5, 70, rng), BOLD)
    vv = np.linspace(-1, 1, 30)
    line(xy(1 - .012 * np.cos(vv * np.pi / 2), vv), BOLD * .85)

    # --- head ------------------------------------------------------------------------------
    vv = np.linspace(-.9, .92, 40)
    gill = xy(.195 + .045 * np.cos(vv * np.pi / 2), vv)
    line(_wobble(gill, .8, 50, rng), BOLD * .9)
    wline(xy(.165 + .04 * np.cos(vv[8:] * np.pi / 2), vv[8:]), THIN, .6, 40)  # preopercle
    for v in np.linspace(.15, .8, 7):  # its serrated edge
        p = xy(.165 + .04 * np.cos(v * np.pi / 2), v)
        line([p, p + 4 * tailv + np.array([0, 1.5])], .8)
    eu, ev = .1, -.22
    re = .26 * (ht(eu) + hb(eu)) / 2
    eye = xy(eu, ev)
    ring(eye, re * 1.25, THIN)            # orbit
    disc(eye, re, tuple(PAPER.astype(int)))
    ring(eye, re, BOLD * .8)
    ring(eye, .62 * re, THIN)
    disc(eye, .45 * re)
    disc(eye + np.array([-.25 * face, -.3]) * re, .14 * re, tuple(PAPER.astype(int)))
    mu = np.linspace(0, .075, 14)
    wline(xy(mu, .02 + 1.4 * mu), 1.6, .3, 30)                 # upper lip
    wline(xy(mu[1:], .2 + 2.6 * mu[1:]), 1.3, .3, 30)          # lower jaw
    wline(xy(np.linspace(.03, .09, 8), .45 + 1.5 * np.linspace(0, .06, 8)), THIN, .3, 30)  # chin fold
    for _ in range(rng.integers(10, 18)):  # cheek dashes, clear of the eye
        p = xy(rng.uniform(.05, .16), rng.uniform(.05, .8))
        if np.linalg.norm(p - eye) > 1.5 * re:
            q = rng.uniform(4, 8) * np.array([face * np.cos(.3), np.sin(.3)])
            line([p, p + q], 1.1)
    for v in np.linspace(-.93, -.6, 4):  # brow contour hatching
        wline(xy(np.linspace(.04 + .06 * (v + 1), .16, 10), v), 1.0, .3, 30)

    # --- fins: thin wavy membranes with many rays (never bold, so never part of the silhouette)
    def outward(us, v):
        tang = xy(us + 1e-3, v) - xy(us - 1e-3, v)
        nrm = np.stack([tang[:, 1], -tang[:, 0]], -1)
        nrm *= np.sign(nrm[:, 1:2]) * np.sign(v)  # up for dorsal, down for ventral
        return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)

    def rays(base, d, lens, curl, gap=0, spiny=False, fil=()):
        q = np.linspace(0, 1, 10)[:, None]
        rs = [inside(b + (gap + q * (ln - gap)) * dd + curl * ln * q ** 2 * tailv) for b, dd, ln in zip(base, d, lens)]
        tips = np.array([b + (.86 if spiny else 1) * (r[-1] - b) for b, r in zip(base, rs)])
        for r in rs:
            wline(r, 1.25 if spiny else 1.05, .5, 35)  # stays well under BOLD: fins must never read as body
        margin = _wobble(_smooth(tips, base, .1), 2.2, 22, rng)  # the wavy fishdraw membrane edge
        line(margin, THIN)
        for b0, b1, t0, t1 in zip(base[:-1], base[1:], tips[:-1], tips[1:]):  # faint striation per gap
            p0, p1 = (b0 + b1) / 2, (t0 + t1) / 2
            line([p0 + .3 * (p1 - p0), p0 + rng.uniform(.5, .75) * (p1 - p0)], .5)
        for i, ext in fil:  # trailing filaments
            r = rs[i]
            dd = (r[-1] - r[-2]) / np.linalg.norm(r[-1] - r[-2])
            q2 = np.linspace(0, 1, 16)[:, None]
            wline(inside(r[-1] + q2 * ext * dd + .6 * ext * q2 ** 2 * tailv), .8, .6, 40)

    def fin(u_a, u_b, v, n, height, profile, tilt, **kw):
        us = np.linspace(u_a, u_b, n)
        tl = tilt * np.linspace(.8, 1.2, n)  # rays lean further back toward the rear of the fin
        d = np.cos(tl)[:, None] * outward(us, v) + np.sin(tl)[:, None] * tailv
        rays(xy(us, v), d, height * profile(np.linspace(0, 1, n)), .1, **kw)

    g = rng.uniform(.5, 1.5)
    if rng.uniform() < .55:  # one big rounded dorsal
        fil = [(0, rng.uniform(.6, 1.2) * DH)] if rng.uniform() < .3 else []
        fin(.3, .72, -1, rng.integers(16, 24), DH, lambda s: .25 + .75 * np.sin(np.pi * s ** g), rng.uniform(.3, .7),
            fil=fil)
    else:                    # spiny + soft dorsal
        fil = [(i, rng.uniform(.4, 1.1) * DH) for i in range(rng.integers(1, 4))] if rng.uniform() < .5 else []
        fin(.3, .5, -1, rng.integers(8, 11), DH, lambda s: .35 + .65 * np.sin(np.pi * s ** .7), .5, spiny=True, fil=fil)
        fin(.54, .76, -1, rng.integers(12, 16), DH * .8, lambda s: .4 + .6 * np.sin(np.pi * s ** g), .7)
    fin(.6, .84, 1, rng.integers(12, 18), VH, lambda s: .35 + .65 * np.sin(np.pi * s ** .8), .6)  # anal
    fin(.3, .38, 1, 7, VH * 1.25, lambda s: .65 + .35 * np.sin(np.pi * s), .7,                    # pelvic
        fil=[(0, rng.uniform(.5, 1.2) * VH)] if rng.uniform() < .3 else [])
    extra = rng.choice(['none', 'adipose', 'finlets'])
    if extra == 'adipose':  # small fleshy fin between dorsal and tail
        us = np.linspace(.8, .88, 14)
        q = np.linspace(0, 1, 14)[:, None]
        wline(xy(us, -1) + outward(us, -1) * rng.uniform(14, 22) * np.sin(np.pi * q) ** .6 + 6 * q ** 2 * tailv,
              THIN, .8, 20)
    elif extra == 'finlets':  # tuna-style flags along the tail stalk
        for v in (-1, 1):
            for u_ in np.linspace(.86, .96, 5):
                b0, b1 = xy(u_, v), xy(u_ + .012, v)
                tip = b0 + outward(np.array([u_]), v)[0] * rng.uniform(10, 15) + 12 * tailv
                line([b0, tip, b1], THIN)

    # caudal: forked, emarginate, truncate or rounded; rays start just off the bold tail base so
    # their converging roots don't blur into a bold blob the decoder would mistake for body
    fork, spread = rng.choice([.5, .25, .05, -.25]) + rng.uniform(-.05, .05), rng.uniform(.5, .8)
    vs = np.linspace(-1, 1, rng.integers(13, 19))
    th = spread * vs
    ext = min(rng.uniform(.25, .5) * tail, (W - L - tail) / 2 - 25)
    rays(xy(np.ones_like(vs), vs), np.stack([face * np.cos(th), np.sin(th)], -1),
         tail * (1 - fork * (1 - np.abs(vs) ** 1.2)), 0, gap=9,
         fil=[(0, ext), (len(vs) - 1, ext)] if fork > .3 and rng.uniform() < .5 else [])

    # --- plate: hairline frame and a binomial ----------------------------------------------
    draw.rectangle([14 * SS, 14 * SS, (W - 14) * SS, (H - 14) * SS], outline=(170, 165, 158), width=1)
    try:
        font = ImageFont.truetype('georgiai.ttf', 22 * SS)
    except OSError:
        font = ImageFont.load_default(22 * SS)
    draw.text((W / 2 * SS, (H - 36) * SS), _name(rng), fill=INK, font=font, anchor='mm')
    return np.asarray(im.resize((W, H), Image.LANCZOS))


# --- decoder -----------------------------------------------------------------------------

def _odd(k):
    return int(k) | 1


def _ink(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) + 1
    k = _odd(max(7, min(g.shape) // 30))  # wider than a blurred line
    return np.clip(1 - g / cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)), 0, 1)


def _body(img):
    """Bold line work -> (body mask, nose tip, tail-base centre)."""
    h, w = img.shape[:2]
    b = cv2.GaussianBlur(_ink(img), (0, 0), max(1., min(h, w) / 400))
    bold = (b > .5 * np.percentile(b, 99.7)).astype(np.uint8)  # thin strokes blur to grey, bold stay dark
    k = _odd(max(3, min(h, w) // 160))
    bold = cv2.morphologyEx(bold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    _, lab = cv2.connectedComponents(1 - bold, connectivity=4)  # fill whatever the bold lines enclose
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    m = (bold | ~np.isin(lab, border)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n < 2:
        return None
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
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
    for _ in range(2):  # nose tip and tail-base centre, pushed out to the extreme along the axis
        s = (P - mu) @ e1
        a, b = s < s.min() + .005 * span, s > s.max() - .01 * span
        nose = P[a].mean(0) + (s.min() - s[a].mean()) * e1
        ped = P[b].mean(0) + (s.max() - s[b].mean()) * e1
        e1 = (ped - nose) / np.linalg.norm(ped - nose)
    return m, nose, ped


def _frame(m, nose, ped):
    """Body frame for a given nose/tail-base: axis e1, e2 toward the belly, length, profile mid/half(u)."""
    L = np.linalg.norm(ped - nose)
    e1 = (ped - nose) / L
    e2 = np.array([-e1[1], e1[0]])
    us, ts = np.linspace(0, 1, 201), np.arange(-.6 * L, .6 * L)
    X = nose[0] + us[:, None] * L * e1[0] + ts * e2[0]
    Y = nose[1] + us[:, None] * L * e1[1] + ts * e2[1]
    prof = cv2.remap(m, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_NEAREST) > 0
    top = np.where(prof.any(1), ts[prof.argmax(1)], 0)
    bot = np.where(prof.any(1), ts[len(ts) - 1 - prof[:, ::-1].argmax(1)], 0)
    top, bot = median_filter(top, 9, mode='nearest'), median_filter(bot, 9, mode='nearest')  # drop spurs from fin roots
    if bot.sum() > -top.sum():  # back is the taller side; make e2 point to the belly
        e2, top, bot = -e2, -bot, -top
    return nose, e1, e2, L, us, (top + bot) / 2, (bot - top) / 2


# templates, in cell units relative to the scale centre: (du offsets, dv offsets), shape (curves, points)
_HT = np.radians(np.linspace(40, 80, 6))
_RR = np.repeat([.45, .6, .75], len(_HT))
HATCH = (np.broadcast_to(-.55 + _RR * np.cos(np.tile(_HT, 3)), (2, len(_RR))),
         np.array([[-1], [1]]) * .5 * _RR * np.sin(np.tile(_HT, 3)))  # dorsal group, ventral group
_AT = np.radians(np.linspace(-80, 80, 9))
ARC = ((-.55 + np.cos(_AT))[None], (.5 * np.sin(_AT))[None])  # the scale outline: same in every scale, bit-free
# local grid tracking per column: shift along u (x du), slant (x du per unit v), shift along v (x dv),
# stretch along v (x dv per unit v)
_COMBOS = np.stack(np.meshgrid(*[np.linspace(-r, r, n) for r, n in [(.35, 7), (.3, 5), (.25, 5), (.3, 5)]],
                               indexing='ij'), -1).reshape(-1, 4)
_COL = CELLS[:, 4].astype(int)
_STARTS = np.searchsorted(_COL, np.arange(COLS + 1))


def _sample(ink, frame, cells, off, tmpl):
    """Mean ink along each template curve of each cell -> (..., n, curves).
    off (..., n|1, 4) nudges cells by the local tracking offsets."""
    nose, e1, e2, L, us, mid, half = frame
    cu, cv, du, dv = (x[:, None, None] for x in cells[:, :4].T)
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
    # 1) global: slide the nose along the perpendicular (its tip is the least certain landmark)
    perp = np.array([-(ped - nose)[1], (ped - nose)[0]]) / L
    frame = max((_frame(m, nose + d * perp, ped) for d in np.linspace(-.03, .03, 13) * L),
                key=lambda f: _sample(ink, f, CELLS, np.zeros(4), ARC).mean())
    # 2) local: perspective / paper curl bend the columns; track each column's shift, slant and
    #    stretch, coarse then fine around the coarse answer
    off = np.zeros((COLS, 4))
    for step in (1, .4):
        off = np.array([o + _COMBOS[np.argmax(_sample(ink, frame, CELLS[a:b], o + step * _COMBOS[:, None], ARC)
                                              .mean((-1, -2)))] * step if b > a else o
                        for o, a, b in zip(off, _STARTS[:-1], _STARTS[1:])])
        off = median_filter(off, size=(3, 1), mode='nearest')  # one noisy column can't drag the grid
    r = _sample(ink, frame, CELLS, off[_COL], HATCH)
    return (r[:CODE_BITS, 0] > r[:CODE_BITS, 1]).astype(np.uint8), []


if __name__ == '__main__':  # round-trip self-check on clean renders (first style attempt only)
    rng = np.random.default_rng(1)
    for z in range(5):
        bits = rng.integers(0, 2, CODE_BITS).astype(np.uint8)
        res = decode(_render(bits, (z, 0)))
        assert res is not None, f'z={z}: not detected'
        assert (res[0] == bits).all(), f'z={z}: {(res[0] != bits).sum()} bit errors'
    print('ok')
