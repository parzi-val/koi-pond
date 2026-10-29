"""Topological baseline: a radial coral whose branch junctions carry the bits.

N spokes leave a central ring; each spoke has K side branches. A bit is which side of the
trunk its branch leaves on, i.e. the cyclic order of edges at that junction, so it survives
any orientation-preserving warp. Spokes are read in cyclic order around the ring, starting
at the one whose tip closes into a loop. One spoke = one codeword byte, so a spoke that
fails to parse becomes an RS erasure rather than shifting every later bit.
"""
import colorsys
import numpy as np, cv2, networkx as nx
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

S, SS = 768, 2       # canvas size, supersampling factor
N, K = 12, 8         # spokes, junctions (bits) per spoke
R, R_TIP = 60, 350   # ring radius, spoke tip radius
J0, DJ = 100, 24     # first junction radius, junction spacing (odd spokes offset by DJ/2)
LOOP = 13            # marker loop radius


def _hsv(h, s, v):
    return tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h % 1, s, v))


def encode(bits, z):
    """Render, then self-verify; re-roll the style (not the bits) if the clean render won't decode."""
    for attempt in range(8):
        img = _render(bits, (z, attempt))
        res = decode(img)
        if res is not None and not res[1] and (res[0] == bits).all():
            return img
    return img  # ponytail: gives up after 8 styles; counts as a clean-decode failure in the harness


def _render(bits, z):
    rng = np.random.default_rng(z)
    h = rng.uniform()
    paper0, paper1 = _hsv(h, rng.uniform(.04, .12), .97), _hsv(h + .05, rng.uniform(.1, .2), .9)
    ink0 = _hsv(h + rng.uniform(.4, .6), rng.uniform(.5, .8), .22)
    ink1 = _hsv(h + rng.uniform(.3, .7), rng.uniform(.6, .9), .38)
    accent = _hsv(h + rng.uniform(.1, .25), rng.uniform(.6, .9), .5)
    base, curl = rng.uniform(0, 2 * np.pi), rng.uniform(-.5, .5)
    # gentle wobble only: tight trunk bends make same-side branches converge and cross
    wob_a, wob_f, phases = rng.uniform(0, .015), rng.uniform(1, 2), rng.uniform(0, 2 * np.pi, N)
    alpha, bend, blen = np.radians(rng.uniform(45, 62)), rng.uniform(-.6, .2), rng.uniform(.17, .22)

    yy, xx = np.mgrid[0:S, 0:S]
    t = np.hypot(xx - S / 2, yy - S / 2)[..., None] / (S / np.sqrt(2))
    bg = (1 - t) * np.array(paper0) + t * np.array(paper1) + rng.normal(0, 2.5, (S, S, 1))
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).resize((S * SS,) * 2, Image.BILINEAR)
    draw = ImageDraw.Draw(img)

    def disc(p, r, col):
        x, y = p * SS
        draw.ellipse([x - r * SS, y - r * SS, x + r * SS, y + r * SS], fill=col)

    def stroke(pts, w0, w1, c0, c1):  # tapered, colour-graded polyline with round joints
        for k, p in enumerate(pts):
            f = k / (len(pts) - 1)
            w = w0 + (w1 - w0) * f
            col = tuple(int(a + (b - a) * f) for a, b in zip(c0, c1))
            disc(p, w / 2, col)
            if k:
                draw.line([*(pts[k - 1] * SS), *(p * SS)], fill=col, width=int(round(w * SS)))

    c = np.array([S / 2, S / 2])

    def trunk(i, r):
        f = (r - R) / (R_TIP - R)
        th = base + 2 * np.pi * i / N + curl * f + wob_a * np.sin(2 * np.pi * wob_f * f + phases[i])
        return c + r[..., None] * np.stack([np.cos(th), np.sin(th)], -1)

    draw.ellipse([*(c - R) * SS, *(c + R) * SS], outline=ink0, width=6 * SS)
    for i in range(N):
        P = trunk(i, np.linspace(R, R_TIP, 160))
        stroke(P, 6, 3, ink0, ink1)
        for j in range(K):
            rj = J0 + DJ * j + DJ / 2 * (i % 2)
            p0, p1 = trunk(i, np.array([rj, rj + 1]))
            sgn = 1 if bits[i * K + j] else -1  # +1: branch rotated +alpha from trunk -> cross(trunk, branch) > 0
            L = min(blen * rj, 36) * rng.uniform(.9, 1.1)
            phi = np.arctan2(*(p1 - p0)[::-1]) + sgn * (alpha + bend * np.linspace(0, 1, 14))
            B = np.vstack([p0, p0 + np.cumsum(np.stack([np.cos(phi), np.sin(phi)], -1) * L / 14, 0)])
            stroke(B, 3.2, 2, ink1, ink1)
            disc(B[-1], 3, accent)
        if i == 0:  # marker: tip closes into a loop
            u = (P[-1] - P[-2]) / np.linalg.norm(P[-1] - P[-2])
            o = P[-1] + u * LOOP
            draw.ellipse([*(o - LOOP) * SS, *(o + LOOP) * SS], outline=ink1, width=int(3.2 * SS))
        else:
            disc(P[-1], 4.5, accent)
    return np.asarray(img.resize((S, S), Image.LANCZOS))


# --- decoder ---------------------------------------------------------------------------

OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _ink_mask(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(7, min(g.shape) // 40) | 1
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ratio = np.clip(255 * g / (bg + 1), 0, 255).astype(np.uint8)  # flattens lighting gradients
    _, ink = cv2.threshold(ratio, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 2:
        return None
    ink = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    # fill pinholes so noise/JPEG speckle doesn't create fake cycles
    _, lab, st, _ = cv2.connectedComponentsWithStats(1 - ink, connectivity=4)
    ink[np.isin(lab, np.nonzero(st[:, cv2.CC_STAT_AREA] < (min(g.shape) / 100) ** 2)[0])] = 1
    return ink


def _graph(sk):
    """Skeleton -> MultiGraph. Nodes: endpoint/junction pixel clusters ('o' = (x, y)).
    Edges: ordered pixel paths ('pts' as (x, y)), 'len', unique 'id'."""
    sk = np.pad(sk, 1).astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    nodes = (sk == 1) & (nb != 2)
    nn, nlab, _, cent = cv2.connectedComponentsWithStats(nodes.astype(np.uint8), connectivity=8)
    G = nx.MultiGraph()
    for v in range(1, nn):
        G.add_node(v, o=cent[v] - 1)
    seen = np.zeros(sk.shape, bool)
    eid = 0
    for y, x in zip(*np.nonzero(nodes)):
        a = nlab[y, x]
        for dy, dx in OFFS:
            cy, cx = y + dy, x + dx
            if not sk[cy, cx] or nlab[cy, cx] or seen[cy, cx]:
                continue
            path, (py, px), end = [(y, x)], (y, x), None
            while end is None:
                path.append((cy, cx))
                seen[cy, cx] = True
                nxt = None
                for ey, ex in OFFS:
                    qy, qx = cy + ey, cx + ex
                    if (qy, qx) == (py, px) or not sk[qy, qx]:
                        continue
                    b = nlab[qy, qx]
                    if b and (b != a or len(path) > 3):
                        end = (b, qy, qx)
                        break
                    if not b and not seen[qy, qx] and nxt is None:
                        nxt = (qy, qx)
                if end is None:
                    if nxt is None:
                        break
                    (py, px), (cy, cx) = (cy, cx), nxt
            if end:
                path.append(end[1:])
                G.add_edge(a, end[0], pts=np.array(path, float)[:, ::-1] - 1, len=len(path), id=eid)
                eid += 1
    return G


def _orient(G, v, d):
    """Edge points ordered starting at node v."""
    p, o = d['pts'], G.nodes[v]['o']
    return p if np.sum((p[0] - o) ** 2) <= np.sum((p[-1] - o) ** 2) else p[::-1]


def _simplify(G, tol):
    """Prune skeleton spurs, dissolve pass-through nodes, merge junctions closer than tol."""
    nid = 10 ** 6
    changed = True
    while changed:
        changed = False
        for v in list(G.nodes):
            if v not in G:
                continue
            es = list(G.edges(v, keys=True, data=True))
            deg = G.degree(v)
            if deg == 0:
                G.remove_node(v)
            elif deg == 1 and es[0][3]['len'] < tol and G.degree(es[0][1]) >= 3:
                G.remove_node(v)
                changed = True
            elif deg == 2 and len(es) == 2:  # not a self-loop
                (_, a, _, d1), (_, b, _, d2) = es
                pts = np.vstack([_orient(G, a, d1), _orient(G, v, d2)])
                G.remove_node(v)
                G.add_edge(a, b, pts=pts, len=d1['len'] + d2['len'], id=nid)
                nid += 1
                changed = True
        if changed:
            continue
        for a, b, key, d in G.edges(keys=True, data=True):
            if a != b and d['len'] < tol and G.degree(a) >= 3 and G.degree(b) >= 3:
                G.remove_edge(a, b, key)
                for _, w, d2 in list(G.edges(b, data=True)):
                    G.add_edge(a, a if w == b else w, **d2)
                G.nodes[a]['o'] = (G.nodes[a]['o'] + G.nodes[b]['o']) / 2
                G.remove_node(b)
                changed = True
                break


def _dir(p):
    return p[max(1, len(p) // 2)] - p[0]


def _walk(G, u, w, d):
    """Follow a spoke outward from ring node u. Returns (bits, is_marker) or None if unparseable."""
    bits, prev, v, inc = [], d['id'], w, _orient(G, u, d)
    for _ in range(2 * K):
        out = [(x, e) for _, x, e in G.edges(v, data=True) if e['id'] != prev]
        if not out:
            return bits, False
        if any(x == v for x, _ in out) or (len(out) == 2 and out[0][0] == out[1][0]):
            return bits, True
        if len(out) != 2:
            return None
        lens = [e['len'] if G.degree(x) == 1 else np.inf for x, e in out]
        if min(lens) == np.inf:
            return None
        s = int(np.argmin(lens))
        a, b = _dir(inc[::-1]), _dir(_orient(G, v, out[s][1]))  # a points back along the trunk
        bits.append(int(a[0] * b[1] - a[1] * b[0] < 0))  # cross(trunk_in, branch) > 0  <=>  bit 1
        nv, e = out[1 - s]
        prev, inc, v = e['id'], _orient(G, v, e), nv
    return None


def decode(img):
    ink = _ink_mask(img)
    if ink is None:
        return None
    sk = skeletonize(ink.astype(bool))
    hw = cv2.distanceTransform(ink, cv2.DIST_L2, 3)[sk]
    G = _graph(sk)
    _simplify(G, max(3.0, 3 * np.median(hw)) if len(hw) else 3.0)
    pos = lambda v: G.nodes[v]['o']
    m = cv2.moments(ink, binaryImage=True)
    centre = (m['m10'] / m['m00'], m['m01'] / m['m00'])
    # ring = smallest basis cycle that encloses the centre of mass
    ring, best = None, np.inf
    for cyc in nx.cycle_basis(nx.Graph(G)):
        if len(cyc) < 3:
            continue
        poly = np.array([pos(v) for v in cyc], np.float32)
        area = cv2.contourArea(poly)
        if area < best and cv2.pointPolygonTest(poly, centre, False) > 0:
            ring, best = cyc, area
    if ring is None:
        return None
    o = np.mean([pos(v) for v in ring], 0)
    spokes = []
    for u in sorted(ring, key=lambda v: np.arctan2(*(pos(v) - o)[::-1])):
        out = [(w, d) for _, w, d in G.edges(u, data=True) if w not in ring]
        if out:
            spokes.append(_walk(G, u, *out[0]) if len(out) == 1 else None)
    marks = [i for i, s in enumerate(spokes) if s and s[1]]
    if len(spokes) != N or len(marks) != 1:
        return None
    spokes = spokes[marks[0]:] + spokes[:marks[0]]
    bits, erased = np.zeros(N * K, np.uint8), []
    for i, s in enumerate(spokes):
        if s and len(s[0]) == K:
            bits[i * K:(i + 1) * K] = s[0]
        else:
            erased.append(i)
    return bits, erased


if __name__ == '__main__':  # round-trip self-check on clean renders
    rng = np.random.default_rng(1)
    for z in range(5):
        bits = rng.integers(0, 2, N * K).astype(np.uint8)
        res = decode(encode(bits, z))
        assert res is not None, f'z={z}: not detected'
        assert not res[1] and (res[0] == bits).all(), f'z={z}: erased={res[1]} errors={(res[0] != bits).sum()}'
    print('ok')
