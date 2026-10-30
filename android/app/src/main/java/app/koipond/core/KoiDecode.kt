package app.koipond.core

import app.koipond.core.KoiSpec.ARC_U
import app.koipond.core.KoiSpec.ARC_V
import app.koipond.core.KoiSpec.CELLS
import app.koipond.core.KoiSpec.CHUNK
import app.koipond.core.KoiSpec.COLS
import app.koipond.core.KoiSpec.COMBOS
import app.koipond.core.KoiSpec.FISH_BITS
import app.koipond.core.KoiSpec.HATCH_U
import app.koipond.core.KoiSpec.HATCH_V
import app.koipond.core.KoiSpec.NF
import app.koipond.core.KoiSpec.STARTS
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/** Port of koi.py's decoder: find each koi's bold silhouette, fit its spine, track its scale grid, read the hatches. */
object KoiDecode {

    /** What the scanner made of one silhouette (for drawing over the photo). */
    class KoiRead(val spine: Pts?, val index: Int?, val mirrored: Boolean)

    class Result(val codeword: ByteArray, val erased: List<Int>, val kois: List<KoiRead>) {
        val read get() = kois.count { it.index != null }
    }

    /** A koi silhouette, cropped: pixels inside (x0, y0, w, h) of the full image. */
    class KMask(val x0: Int, val y0: Int, val w: Int, val h: Int, val a: ByteArray) {
        fun at(x: Int, y: Int): Boolean {
            val xx = x - x0
            val yy = y - y0
            return xx >= 0 && yy >= 0 && xx < w && yy < h && a[yy * w + xx].toInt() != 0
        }
    }

    private fun odd(k: Int) = k or 1

    fun inkMap(img: Rgb): FImg {
        val v = FImg(img.w, img.h, FloatArray(img.px.size) {
            val p = img.px[it]
            (max(max((p shr 16) and 0xff, (p shr 8) and 0xff), p and 0xff) + 1).toFloat()  // value channel: pale washes vanish
        })
        val c = closeRect(v, odd(max(7, min(img.w, img.h) / 80)))
        return FImg(img.w, img.h, FloatArray(v.a.size) { (1 - v.a[it] / c.a[it]).coerceIn(0f, 1f) })
    }

    private fun binMorph(m: ByteArray, w: Int, h: Int, offs: Offs, dilate: Boolean): ByteArray {
        val out = ByteArray(m.size)
        for (y in 0 until h) for (x in 0 until w) {
            var v = !dilate
            for (o in 0 until offs.n) {
                val xx = x + offs.dx[o]
                val yy = y + offs.dy[o]
                val inside = xx in 0 until w && yy in 0 until h
                val s = inside && m[yy * w + xx].toInt() != 0
                if (dilate && s) { v = true; break }
                if (!dilate && inside && !s) { v = false; break }
            }
            out[y * w + x] = if (v) 1 else 0
        }
        return out
    }

    /** Stroke-erasing kernel: sized from the drawing (99th-percentile stroke half-width), measured on the
     *  central half of the image (widths are the same everywhere; this keeps the distance transform cheap). */
    private fun strokeKernel(ink: FImg): Int {
        val cw = ink.w / 2
        val ch = ink.h / 2
        val x0 = ink.w / 4
        val y0 = ink.h / 4
        val crop = ByteArray(cw * ch) { if (ink.a[(y0 + it / cw) * ink.w + x0 + it % cw] > .5f) 1 else 0 }
        val dt = edt(crop, cw, ch)
        val maxDt = dt.a.maxOrNull() ?: 0f
        return if (maxDt > 0) min(8, (1.6 * percentile(dt.a, 99.0, maxDt + 1) { it > 0 }).toInt()) else 0
    }

    /** Bold strokes -> filled silhouettes, one per koi. Thin strokes are erased at full resolution (where the
     *  outline is still clearly the boldest line); big images then search for silhouettes at half resolution. */
    fun koiMasks(ink: FImg, fullRes: Boolean = false): List<KMask> {
        val ko = Prof.sec("m.edt") { strokeKernel(ink) }
        val opened = Prof.sec("m.open") { if (ko >= 2) openGray(ink, ellipseOffsets(ko)) else ink }
        if (fullRes || min(ink.w, ink.h) < 1600) return koiMasksAt(opened)
        return koiMasksAt(down2(opened)).map { m ->
            val w = min(ink.w - 2 * m.x0, 2 * m.w)
            val h = min(ink.h - 2 * m.y0, 2 * m.h)
            KMask(2 * m.x0, 2 * m.y0, w, h, ByteArray(w * h) { m.a[(it / w / 2) * m.w + (it % w) / 2] })
        }
    }

    private fun above(a: FloatArray, t: Float) = ByteArray(a.size) { if (a[it] > t) 1 else 0 }

    /** Everything the bold lines enclose, plus the lines themselves. */
    private fun fillHoles(bold: ByteArray, w: Int, h: Int): ByteArray {
        val (lab, _) = components(ByteArray(w * h) { if (bold[it].toInt() == 0) 1 else 0 }, w, h)
        val border = HashSet<Int>()
        for (x in 0 until w) { border += lab[x]; border += lab[(h - 1) * w + x] }
        for (y in 0 until h) { border += lab[y * w]; border += lab[y * w + w - 1] }
        val isBorder = BooleanArray(lab.maxOrNull()!! + 1)
        for (l in border) isBorder[l] = true
        return ByteArray(w * h) { if (bold[it].toInt() != 0 || !isBorder[lab[it]]) 1 else 0 }
    }

    /** Binary opening with a disc of radius rad, via two distance transforms (fast for big discs). */
    private fun openDisc(m: ByteArray, w: Int, h: Int, rad: Double): ByteArray {
        val er = edt(m, w, h)
        val eroded = ByteArray(w * h) { if (er.a[it] > rad) 0 else 1 }  // 1 = outside the eroded set
        val back = edt(eroded, w, h)
        return ByteArray(w * h) { if (back.a[it] <= rad) 1 else 0 }
    }

    private class Stats(val area: IntArray, val x0: IntArray, val y0: IntArray, val x1: IntArray, val y1: IntArray)

    private fun stats(lab: IntArray, n: Int, w: Int): Stats {
        val s = Stats(IntArray(n), IntArray(n) { Int.MAX_VALUE }, IntArray(n) { Int.MAX_VALUE }, IntArray(n) { -1 }, IntArray(n) { -1 })
        for (i in lab.indices) {
            val l = lab[i]
            if (l == 0) continue
            s.area[l]++
            val x = i % w
            val y = i / w
            if (x < s.x0[l]) s.x0[l] = x
            if (y < s.y0[l]) s.y0[l] = y
            if (x > s.x1[l]) s.x1[l] = x
            if (y > s.y1[l]) s.y1[l] = y
        }
        return s
    }

    /** The silhouette is the unopened mask within 2 px of the chosen component. */
    private fun cut(lab: IntArray, l: Int, raw: ByteArray, w: Int, h: Int, st: Stats): KMask {
        val x0 = max(0, st.x0[l] - 3)
        val y0 = max(0, st.y0[l] - 3)
        val cw = min(w - 1, st.x1[l] + 3) - x0 + 1
        val ch = min(h - 1, st.y1[l] + 3) - y0 + 1
        val sub = FImg(cw, ch, FloatArray(cw * ch) { if (lab[(y0 + it / cw) * w + x0 + it % cw] == l) 1f else 0f })
        val dil = rectFilter(sub, 5, true)
        return KMask(x0, y0, cw, ch, ByteArray(cw * ch) {
            if (dil.a[it] > 0 && raw[(y0 + it / cw) * w + x0 + it % cw].toInt() != 0) 1 else 0
        })
    }

    private fun koiMasksAt(ink: FImg): List<KMask> {
        val w = ink.w
        val h = ink.h
        val b = Prof.sec("m.blur") { gaussBlur(ink, max(1.0, min(w, h) / 1500.0)) }
        val ell = ellipseOffsets(odd(max(3, min(w, h) / 600)))
        val k = ell.dx.maxOrNull()!! * 2 + 1
        val bold = Prof.sec("m.close") { binMorph(binMorph(above(b.a, .5f * percentile(b.a, 99.7, 1f)), w, h, ell, true), w, h, ell, false) }
        val raw = Prof.sec("m.fill") { fillHoles(bold, w, h) }
        val (lab, n) = Prof.sec("m.opendisc") { components(openDisc(raw, w, h, k + .5), w, h) }
        val st = stats(lab, n, w)
        // a koi never touches the frame; anything that does is border smear or clutter
        val score = IntArray(n) { l -> if (l > 0 && st.x0[l] > 0 && st.y0[l] > 0 && st.x1[l] < w - 1 && st.y1[l] < h - 1) st.area[l] else 0 }
        val best = score.maxOrNull() ?: 0
        if (best == 0) return emptyList()
        return Prof.sec("m.cut") { (1 until n).filter { score[it] > max(.35 * best, 1e-3 * w * h) }.map { cut(lab, it, raw, w, h, st) } }
    }

    class Frame(val sp: Pts, val nrm: Pts, val len: Double, val mid: DoubleArray, val half: DoubleArray) {
        private val n = sp.n - 1
        fun at(a: DoubleArray, u: Double): Double {
            val t = u.coerceIn(0.0, 1.0) * n
            val i = min(n - 1, t.toInt())
            val f = t - i
            return a[i] + f * (a[i + 1] - a[i])
        }
        /** Image point of body coordinate (u, v); one lookup for both axes. */
        fun sample(ink: FImg, u: Double, v: Double): Float {
            val t = u.coerceIn(0.0, 1.0) * n
            val i = min(n - 1, t.toInt())
            val f = t - i
            fun l(a: DoubleArray) = a[i] + f * (a[i + 1] - a[i])
            val off = l(mid) + v * l(half)
            return sampleBilinear(ink, l(sp.x) + off * l(nrm.x), l(sp.y) + off * l(nrm.y))
        }
    }

    fun frame(m: KMask, sp: Pts): Frame {
        val len = cumlen(sp).last()
        val g = gradient(sp)
        val tg = Pts.of(sp.n) { P(g.x[it], g.y[it]).unit() }
        val nrm = Pts.of(sp.n) { P(-tg.y[it], tg.x[it]) }
        val t0 = -.3 * len
        val nt = kotlin.math.ceil(.6 * len).toInt()
        val lo = DoubleArray(sp.n)
        val hi = DoubleArray(sp.n)
        for (i in 0 until sp.n) {
            var first = -1
            var last = -1
            for (j in 0 until nt) {
                val t = t0 + j
                if (t >= .3 * len) break
                if (m.at((sp.x[i] + nrm.x[i] * t).roundToInt(), (sp.y[i] + nrm.y[i] * t).roundToInt())) {
                    if (first < 0) first = j
                    last = j
                }
            }
            lo[i] = if (first >= 0) t0 + first else 0.0
            hi[i] = if (last >= 0) t0 + last else 0.0
        }
        val l = median1d(lo, 15)
        val hh = median1d(hi, 15)  // fin-root bumps
        return Frame(sp, nrm, len, DoubleArray(sp.n) { (l[it] + hh[it]) / 2 }, DoubleArray(sp.n) { (hh[it] - l[it]) / 2 })
    }

    /** Silhouette -> 201-point centreline from one end to the other (direction not yet known). */
    fun spine(m: KMask): Pts? {
        var sx = 0.0
        var sy = 0.0
        var cnt = 0
        for (y in 0 until m.h) for (x in 0 until m.w) if (m.a[y * m.w + x].toInt() != 0) { sx += x + m.x0; sy += y + m.y0; cnt++ }
        if (cnt < 100) return null
        val mx = sx / cnt
        val my = sy / cnt
        var cxx = 0.0
        var cxy = 0.0
        var cyy = 0.0
        for (y in 0 until m.h) for (x in 0 until m.w) if (m.a[y * m.w + x].toInt() != 0) {
            val dx = x + m.x0 - mx
            val dy = y + m.y0 - my
            cxx += dx * dx; cxy += dx * dy; cyy += dy * dy
        }
        val th = .5 * atan2(2 * cxy, cxx - cyy)
        val e1 = P(cos(th), sin(th))
        val e2 = P(-e1.y, e1.x)
        val ss = DoubleArray(cnt)
        val ts = DoubleArray(cnt)
        var q = 0
        for (y in 0 until m.h) for (x in 0 until m.w) if (m.a[y * m.w + x].toInt() != 0) {
            val dx = x + m.x0 - mx
            val dy = y + m.y0 - my
            ss[q] = dx * e1.x + dy * e1.y
            ts[q] = dx * e2.x + dy * e2.y
            q++
        }
        val smin = ss.min()
        val smax = ss.max()
        val edges = linspace(smin, smax, 41)
        val tmin = DoubleArray(40) { Double.MAX_VALUE }
        val tmax = DoubleArray(40) { -Double.MAX_VALUE }
        val bc = IntArray(40)
        for (i in 0 until cnt) {
            val bi = ((ss[i] - smin) / (smax - smin) * 40).toInt()
            if (bi >= 40) continue  // the right edge itself is outside the last half-open bin
            bc[bi]++
            tmin[bi] = min(tmin[bi], ts[i]); tmax[bi] = max(tmax[bi], ts[i])
        }
        val cs = ArrayList<Double>()
        val mids = ArrayList<Double>()
        for (bi in 0 until 40) if (bc[bi] > 5) { cs += (edges[bi] + edges[bi + 1]) / 2; mids += (tmin[bi] + tmax[bi]) / 2 }
        if (cs.size < 10) return null
        val co = polyfit3(cs.subList(2, cs.size - 2), mids.subList(2, mids.size - 2))
        val sg = linspace(smin, smax, 300)
        val pts = Pts.of(300) {
            val s = sg[it]
            val t = ((co[0] * s + co[1]) * s + co[2]) * s + co[3]
            P(mx + s * e1.x + t * e2.x, my + s * e1.y + t * e2.y)
        }
        fun march(p0: P, d0: P): P {
            val d = d0.unit()
            var p = p0
            repeat(600) {
                if (!m.at((p.x + d.x).roundToInt(), (p.y + d.y).roundToInt())) return p
                p = p + d
            }
            return p
        }
        fun extend(sp: Pts, a: Int): Pts {  // re-find both tips by marching out along the local tangent near each end
            val b = sp.n - 1 - a
            val head = march(sp[a], sp[a] - sp[2 * a])
            val tail = march(sp[b], sp[b] - sp[b - a])
            return resampleN(Pts.of(head) + sp.sub(a, b + 1) + Pts.of(tail), 201)
        }
        // the cubic is only a first guess; converge onto the midline measured along the spine's own normals
        var sp = extend(pts, 20)
        repeat(3) {
            val fr = frame(m, sp)
            val mid = gauss1d(fr.mid, 3.0)
            sp = extend(Pts.of(sp.n) { sp[it] + fr.nrm[it] * mid[it] }, 14)
        }
        return sp
    }

    /** Least-squares cubic, highest power first (numpy.polyfit order). */
    private fun polyfit3(xs: List<Double>, ys: List<Double>): DoubleArray {
        val sc = xs.maxOf { kotlin.math.abs(it) }.coerceAtLeast(1.0)
        val a = Array(4) { DoubleArray(5) }
        for (i in xs.indices) {
            val x = xs[i] / sc
            val pw = doubleArrayOf(x * x * x, x * x, x, 1.0)
            for (r in 0..3) { for (c in 0..3) a[r][c] += pw[r] * pw[c]; a[r][4] += pw[r] * ys[i] }
        }
        for (c in 0..3) {  // Gaussian elimination with partial pivoting
            val p = (c..3).maxBy { kotlin.math.abs(a[it][c]) }
            val t = a[c]; a[c] = a[p]; a[p] = t
            for (r in 0..3) if (r != c) {
                val f = a[r][c] / a[c][c]
                for (k in c..4) a[r][k] -= f * a[c][k]
            }
        }
        val co = DoubleArray(4) { a[it][4] / a[it][it] }
        return doubleArrayOf(co[0] / (sc * sc * sc), co[1] / (sc * sc), co[2] / sc, co[3])
    }

    /** Mean ink along one template curve of cell k, with the cell's tracking offset. */
    private fun curve(ink: FImg, fr: Frame, k: Int, o: DoubleArray, tu: DoubleArray, tv: DoubleArray, sign: Double, from: Int, to: Int): Double {
        val c = CELLS[k]
        var acc = 0.0
        for (i in from until to) {
            val u = c.u + (o[0] + o[1] * c.v + tu[i]) * c.du
            val v = c.v + (o[2] + o[3] * c.v + sign * tv[i]) * c.dv
            acc += fr.sample(ink, u, v)
        }
        return acc / (to - from)
    }

    /** Spine and frame running nose -> tail. Head end: koi are broad-shouldered and taper hard to the tail stalk. */
    private fun headFrame(m: KMask): Frame? {
        val sp = Prof.sec("d.spine") { spine(m) } ?: return null
        return listOf(frame(m, sp), frame(m, sp.reversed())).maxBy { f -> f.half.copyOfRange(10, 41).average() - f.half.copyOfRange(160, 191).average() }
    }

    /** The first candidate bit set (canonical order) that passes the inner RS -> koi index and chunk. */
    private fun inner(fr: Frame, candidates: List<Pair<Boolean, IntArray>>): Pair<KoiRead, ByteArray?> {
        for ((mir, bits) in candidates) {
            val bytes = ByteArray(FISH_BITS / 8) { i -> var v = 0; for (j in 0..7) v = (v shl 1) or bits[i * 8 + j]; v.toByte() }
            val msg = try { Rs.decode(bytes, KoiSpec.INNER_NSYM) } catch (e: RsException) { continue }
            val head = msg[0].toInt() and 0xff
            if (head shr 4 == 0xA && (head and 15) < NF) return KoiRead(fr.sp, head and 15, mir) to msg.copyOfRange(1, msg.size)
        }
        return KoiRead(fr.sp, null, false) to null
    }

    private fun readKoi(ink: FImg, fr: Frame): Pair<KoiRead, ByteArray?> {
        var off = Array(COLS) { DoubleArray(4) }
        for (step in doubleArrayOf(1.0, .4)) Prof.sec("d.tracking") {  // local tracking: shift, slant, stretch per column, coarse then fine
            off = Array(COLS) { col ->
                val a = STARTS[col]
                val b = STARTS[col + 1]
                if (b <= a) return@Array off[col]
                var bestScore = -1.0
                var best = off[col]
                for (cmb in COMBOS) {
                    val o = DoubleArray(4) { off[col][it] + step * cmb[it] }
                    var s = 0.0
                    for (kk in a until b) s += curve(ink, fr, kk, o, ARC_U, ARC_V, 1.0, 0, 9)
                    if (s > bestScore) { bestScore = s; best = o }
                }
                best
            }
            val med = Array(4) { p -> median1d(DoubleArray(COLS) { off[it][p] }, 3) }
            off = Array(COLS) { c -> DoubleArray(4) { med[it][c] } }
        }
        val canon = IntArray(CELLS.size) { k ->
            val o = off[CELLS[k].col]
            val dorsal = curve(ink, fr, k, o, HATCH_U, HATCH_V, -1.0, 0, 18)
            val ventral = curve(ink, fr, k, o, HATCH_U, HATCH_V, 1.0, 0, 18)
            if (dorsal > ventral) 1 else 0
        }
        return inner(fr, listOf(false to canon, true to IntArray(CELLS.size) { 1 - canon[KoiSpec.MIRROR[it]] }))
    }

    /** fullRes: look for silhouettes at full resolution too (slower; a fallback for hard photos). */
    fun decode(img: Rgb, fullRes: Boolean = false, onKoi: (List<KoiRead>, Int) -> Unit = { _, _ -> }): Result? {
        val ink0 = Prof.sec("d.ink") { inkMap(img) }
        val masks = Prof.sec("d.masks") { koiMasks(ink0, fullRes) }
        if (masks.isEmpty()) return null
        val sigma = max(.7, .05 * CELLS[0].du * (KoiSpec.L_RNG.first + KoiSpec.L_RNG.second) / 2 * sqrt((img.w * img.h).toDouble()) / KoiSpec.S)
        val ink = Prof.sec("d.blur") { gaussBlur(ink0, sigma) }
        val frames = masks.map { headFrame(it) }
        // the learned reader (when loaded) reads every koi in one batch; otherwise track and read the hatches
        val learned = if (StripReader.ready && frames.any { it != null })
            Prof.sec("d.reader") { StripReader.read(ink, frames.filterNotNull()) }.iterator() else null
        val chunks = HashMap<Int, ByteArray>()
        val reads = ArrayList<KoiRead>()
        for (fr in frames) {  // report after every koi, so a UI can show them being read one by one
            val (r, chunk) = when {
                fr == null -> KoiRead(null, null, false) to null
                learned != null -> learned.next().let { (asIs, flipped) -> inner(fr, listOf(false to asIs, true to flipped)) }
                else -> readKoi(ink, fr)
            }
            if (r.index != null && chunk != null) chunks.putIfAbsent(r.index, chunk)
            reads += r
            onKoi(reads.toList(), masks.size)
        }
        val code = ByteArray(NF * CHUNK)
        val erased = ArrayList<Int>()
        for (f in 0 until NF) {
            val ch = chunks[f]
            for (i in 0 until CHUNK) if (ch != null) code[f + i * NF] = ch[i] else erased += f + i * NF
        }
        return Result(code, erased.sorted(), reads)
    }
}
