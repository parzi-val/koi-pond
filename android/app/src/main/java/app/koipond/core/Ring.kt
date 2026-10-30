package app.koipond.core

import app.koipond.core.KoiDecode.KMask
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/** Port of koi.py's _ring_masks: silhouettes on an unwarped pond from the layout instead of closed outlines. Every koi
 *  swims the ring, so unroll the band around it and, per angle, take the innermost and outermost bold pixel as the two
 *  flanks; a koi is a long run of angles with a body-wide gap between them, and a broken outline costs a few columns. */
object Ring {
    private const val RING_LO = 845.0  // spine radius of every koi (R0 870..910, +-25 per koi), canvas px
    private const val RING_HI = 935.0

    fun masks(bold: ByteArray, n: Int): List<KMask> {
        val g = n.toDouble() / KoiSpec.S
        val c = n / 2.0
        val r0 = ((RING_LO - 12 - KoiSpec.HW_RNG.second - 10) * g).toInt()
        val nr = ((RING_HI + 12 + KoiSpec.HW_RNG.second + 10) * g).toInt() - r0
        val nth = (2 * PI * (RING_LO + RING_HI) / 2 * g).toInt()
        val fb = FImg(n, n, FloatArray(n * n) { bold[it].toFloat() })
        val seen = BooleanArray(nth)
        var lo = DoubleArray(nth)
        var hi = DoubleArray(nth)
        for (j in 0 until nth) {  // cv2.remap INTER_LINEAR > .5, then first / last bold row
            val t = 2 * PI * j / nth
            for (i in 0 until nr) {
                if (sampleBilinear(fb, c + (r0 + i) * cos(t), c + (r0 + i) * sin(t)) > .5f) {
                    if (!seen[j]) lo[j] = i.toDouble()
                    seen[j] = true
                    hi[j] = i.toDouble()
                }
            }
        }
        fun span(x: Double) = (x * nth / (2 * PI)).toInt() or 1
        // fin roots and pads push a flank outward, outline gaps let it fall in onto the scales, each over a narrow
        // span of angle; a koi's real flank is smooth: broad median, then a running mean (koi.py _flanks)
        lo = meanWrap(medianWrap(lo, span(.25)), span(.1))
        hi = meanWrap(medianWrap(hi, span(.25)), span(.1))
        val mid = DoubleArray(nth) { r0 + (lo[it] + hi[it]) / 2 }
        val wid = DoubleArray(nth) { hi[it] - lo[it] }
        var body = DoubleArray(nth) {
            if (wid[it] > 2 * 35 * g && wid[it] < 2 * (KoiSpec.HW_RNG.second + 15) * g && mid[it] > (RING_LO - 25) * g && mid[it] < (RING_HI + 25) * g) 1.0 else 0.0
        }
        val w = span(.1)  // bridge short breaks (a fin or pad bulging a flank); too-long runs get split below
        body = filt(filt(body, w, true, wrap = true), w, false, wrap = true)
        val isBody = BooleanArray(nth) { body[it] > 0 }
        if (isBody.all { it } || isBody.none { it }) return emptyList()
        val start = isBody.indexOfFirst { !it }  // unroll from a gap, so no koi straddles the seam
        val idx = IntArray(nth) { (it + start) % nth }
        val out = ArrayList<KMask>()
        var j = 0
        while (j < nth) {
            if (!isBody[idx[j]]) { j++; continue }
            var k = j
            while (k < nth && isBody[idx[k]]) k++
            // koi swimming nose to tail can merge into one run: split it where the body is narrowest
            val runs = ArrayDeque(listOf(Run(j, k - 1, true, true)))
            val done = ArrayList<Run>()
            val m0 = (.3 * nth / (2 * PI)).toInt()
            while (runs.isNotEmpty()) {
                val q = runs.removeLast()
                if ((q.b - q.a + 1) * 2 * PI / nth > 1.1 && q.b - q.a > 2 * m0) {
                    var cut = q.a + m0
                    for (x in q.a + m0 until q.b - m0) if (wid[idx[x]] < wid[idx[cut]]) cut = x
                    runs += Run(q.a, cut - 1, q.ea, false)
                    runs += Run(cut + 1, q.b, false, q.eb)
                } else done += q
            }
            for (q in done) {
                var a = q.a
                var b = q.b
                for ((step, ext) in listOf(-1 to q.ea, 1 to q.eb)) {  // out to the snout tip / tail base: narrower, same line
                    var e = if (step < 0) a else b
                    if (ext) for (unused in 0 until (.15 * nth / (2 * PI)).toInt()) {
                        val p = idx[Math.floorMod(e + step, nth)]
                        if (!(seen[p] && 6 * g < wid[p] && abs(mid[p] - mid[idx[Math.floorMod(e, nth)]]) < 30 * g)) break
                        e += step
                    }
                    if (step < 0) a = e else b = e
                }
                if ((b - a + 1) * 2 * PI / nth in .45..1.1) out += silhouette(a, b, idx, lo, hi, r0, nth, n, c)
            }
            j = k
        }
        return out
    }

    private class Run(val a: Int, val b: Int, val ea: Boolean, val eb: Boolean)

    /** One run of columns -> its silhouette. A fin root bulges a flank over a narrow span of angle, snout and tail
     *  taper over a long one: grey closing of the inner flank / opening of the outer flank along the angle. */
    private fun silhouette(a: Int, b: Int, idx: IntArray, lo: DoubleArray, hi: DoubleArray, r0: Int, nth: Int, n: Int, c: Double): KMask {
        val cols = IntArray(b - a + 1) { idx[Math.floorMod(a + it, nth)] }
        val l = (.08 * nth / (2 * PI)).toInt() or 1
        val rl = filt(filt(DoubleArray(cols.size) { lo[cols[it]] }, l, true), l, false)
        val rh = filt(filt(DoubleArray(cols.size) { hi[cols[it]] }, l, false), l, true)
        val poly = Pts.of(2 * cols.size) {
            val i = if (it < cols.size) it else 2 * cols.size - 1 - it
            val r = r0 + if (it < cols.size) rl[i] else rh[i]
            val t = 2 * PI * cols[i] / nth
            P(c + r * cos(t), c + r * sin(t))
        }
        val x0 = max(0, floor(poly.x.min()).toInt() - 1)
        val y0 = max(0, floor(poly.y.min()).toInt() - 1)
        val x1 = min(n - 1, ceil(poly.x.max()).toInt() + 1)
        val y1 = min(n - 1, ceil(poly.y.max()).toInt() + 1)
        val m = Mask(x1 - x0 + 1, y1 - y0 + 1)
        m.fillPoly(Pts.of(poly.n) { P(poly.x[it] - x0, poly.y[it] - y0) })
        return KMask(x0, y0, m.w, m.h, ByteArray(m.a.size) { if (m.a[it].toInt() != 0) 1 else 0 })
    }

    /** Sliding max (or min) over `size` samples: wrap-around, or clamped at the ends (scipy mode 'nearest'). */
    private fun filt(a: DoubleArray, size: Int, isMax: Boolean, wrap: Boolean = false): DoubleArray {
        val h = size / 2
        val n = a.size
        return DoubleArray(n) { i ->
            var v = if (isMax) -Double.MAX_VALUE else Double.MAX_VALUE
            for (t in i - h..i + h) {
                val x = a[if (wrap) Math.floorMod(t, n) else t.coerceIn(0, n - 1)]
                v = if (isMax) max(v, x) else min(v, x)
            }
            v
        }
    }

    private fun medianWrap(a: DoubleArray, size: Int): DoubleArray {
        val h = size / 2
        val win = DoubleArray(size)
        return DoubleArray(a.size) { i ->
            for (t in 0 until size) win[t] = a[Math.floorMod(i + t - h, a.size)]
            win.sort()
            win[h]
        }
    }

    private fun meanWrap(a: DoubleArray, size: Int): DoubleArray {
        val h = size / 2
        return DoubleArray(a.size) { i -> var s = 0.0; for (t in i - h..i + h) s += a[Math.floorMod(t, a.size)]; s / size }
    }
}
