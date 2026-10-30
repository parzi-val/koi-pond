package app.koipond.core

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Port of koi.py's _anchors / _rectify: find the corner bullseyes, unwarp the pond onto a RECT x RECT canvas. */
object Anchors {

    /** The unwarped pond, and the map from its pixels back to the photo's (for drawing over the photo). */
    class Rectified(val img: Rgb, val toPhoto: (P) -> P)

    /** Corner bullseye centres in photo pixels, ordered TL, TR, BR, BL up to rotation; null when fewer than 3 are seen. */
    fun find(img: Rgb): List<P>? {
        val k = max(1, ceil(max(img.w, img.h) / 1000.0).toInt())  // box-downsample to ~1000 px
        val w = img.w / k
        val h = img.h / k
        val small = Rgb(w, h, IntArray(w * h) { i ->
            val x0 = (i % w) * k
            val y0 = (i / w) * k
            var r = 0; var g = 0; var b = 0
            for (y in y0 until y0 + k) for (x in x0 until x0 + k) {
                val p = img.px[y * img.w + x]
                r += (p shr 16) and 255; g += (p shr 8) and 255; b += p and 255
            }
            val n = k * k
            ((r / n) shl 16) or ((g / n) shl 8) or (b / n)
        })
        val ink = KoiDecode.inkMap(small, min(w, h) / 25 or 1)  // wider than the centre dot, or it reads hollow
        val bin = ByteArray(w * h) { if (ink.a[it] > .3f) 1 else 0 }
        val (lab, n) = components(bin, w, h, eight = true)
        val x0 = IntArray(n) { Int.MAX_VALUE }; val y0 = IntArray(n) { Int.MAX_VALUE }
        val x1 = IntArray(n) { -1 }; val y1 = IntArray(n) { -1 }; val area = IntArray(n)
        for (i in lab.indices) {
            val l = lab[i]
            if (l == 0) continue
            val x = i % w; val y = i / w
            area[l]++
            x0[l] = min(x0[l], x); y0[l] = min(y0[l], y); x1[l] = max(x1[l], x); y1[l] = max(y1[l], y)
        }
        val found = ArrayList<Pair<P, Double>>()  // centre (small px), size
        for (l in 1 until n) {
            val bw = x1[l] - x0[l] + 1
            val bh = y1[l] - y0[l] + 1
            if (min(bw, bh) < 10 || bw.toDouble() / bh !in .5..2.0 || area[l] < .15 * bw * bh) continue
            // the ring's hole: everything in its padded box not reachable from outside without crossing it
            val cw = bw + 2
            val ch = bh + 2
            val notRing = ByteArray(cw * ch) { j ->
                val x = x0[l] - 1 + j % cw; val y = y0[l] - 1 + j / cw
                if (x in x0[l]..x1[l] && y in y0[l]..y1[l] && lab[y * w + x] == l) 0 else 1
            }
            val cl = components(notRing, cw, ch).first
            val outside = cl[0]
            var filled = 0; var hole = 0; var dot = 0
            var sx = 0.0; var sy = 0.0; var dx = 0.0; var dy = 0.0
            for (j in 0 until cw * ch) {
                val ring = notRing[j].toInt() == 0
                val inHole = !ring && cl[j] != outside
                if (!ring && !inHole) continue
                val x = x0[l] - 1 + j % cw; val y = y0[l] - 1 + j / cw
                filled++; sx += x; sy += y
                if (inHole) {
                    hole++
                    if (bin[y * w + x].toInt() != 0) { dot++; dx += x; dy += y }
                }
            }
            if (hole.toDouble() / filled !in .25..0.85 || filled < .6 * Math.PI / 4 * bw * bh) continue
            if (dot.toDouble() / filled !in .03..0.3) continue
            val c = P(sx / filled, sy / filled)
            if (hypot(c.x - dx / dot, c.y - dy / dot) > .2 * max(bw, bh)) continue
            found += c to sqrt(filled.toDouble())
        }
        lastFound = found.size
        if (found.size < 3) return null
        val top = found.sortedBy { -it.second }.take(8)
        var best = 0.0
        var quad: List<P>? = null
        for (kk in intArrayOf(4, 3)) {
            for (combo in combos(top.size, kk)) {
                val sz = combo.map { top[it].second }
                if (sz.max() > 2.5 * sz.min()) continue
                var pts = combo.map { top[it].first }
                if (kk == 3) {  // one corner lost: the right-angle vertex is opposite the longest side; complete the parallelogram
                    val d = (0..2).map { j -> dist(pts[(j + 1) % 3], pts[(j + 2) % 3]) }
                    val j = d.indices.maxBy { d[it] }
                    val a = pts[(j + 1) % 3]; val b = pts[(j + 2) % 3]; val c = pts[j]
                    pts = pts + P(a.x + b.x - c.x, a.y + b.y - c.y)
                }
                val ordered = byAngle(pts)
                val a = convexArea(ordered) ?: continue
                if (!squareish(ordered)) continue
                if (a > best) { best = a; quad = ordered }
            }
            if (quad != null) break
        }
        if (quad == null || best < .04 * w * h) return null
        return quad.map { P((it.x + .5) * k - .5, (it.y + .5) * k - .5) }
    }

    /** How many bullseyes the last find() saw (for the lab log). */
    @Volatile var lastFound = 0

    fun rectify(img: Rgb): Rectified? {
        val quad = find(img) ?: return null
        val f = KoiSpec.RECT.toDouble() / KoiSpec.S
        val H = homography(KoiSpec.ANCHORS.map { P(it.x * f, it.y * f) }, quad)
        val toPhoto = { p: P -> val z = H[6] * p.x + H[7] * p.y + 1; P((H[0] * p.x + H[1] * p.y + H[2]) / z, (H[3] * p.x + H[4] * p.y + H[5]) / z) }
        val n = KoiSpec.RECT
        val out = IntArray(n * n)
        for (y in 0 until n) for (x in 0 until n) {
            val z = H[6] * x + H[7] * y + 1
            out[y * n + x] = bilinear(img, (H[0] * x + H[1] * y + H[2]) / z, (H[3] * x + H[4] * y + H[5]) / z)
        }
        // blank the anchors (koi never come near the corners), with the pond's water colour (the median pixel)
        val c = ((KoiSpec.ANCHOR_M + KoiSpec.ANCHOR_R + 20) * f).toInt()
        val water = IntArray(3) { ch -> val v = IntArray(out.size / 97) { i -> (out[97 * i] shr (16 - 8 * ch)) and 255 }; v.sort(); v[v.size / 2] }
        val wc = (water[0] shl 16) or (water[1] shl 8) or water[2]
        for ((ya, xa) in listOf(0 to 0, 0 to n - c, n - c to 0, n - c to n - c)) for (y in ya until ya + c) for (x in xa until xa + c) out[y * n + x] = wc
        return Rectified(Rgb(n, n, out), toPhoto)
    }

    private fun bilinear(img: Rgb, x: Double, y: Double): Int {  // replicate border, like BORDER_REPLICATE
        val xc = x.coerceIn(0.0, img.w - 1.0)
        val yc = y.coerceIn(0.0, img.h - 1.0)
        val xa = min(img.w - 2, floor(xc).toInt()).coerceAtLeast(0)
        val ya = min(img.h - 2, floor(yc).toInt()).coerceAtLeast(0)
        val fx = xc - xa
        val fy = yc - ya
        val p00 = img.px[ya * img.w + xa]; val p10 = img.px[ya * img.w + xa + 1]
        val p01 = img.px[(ya + 1) * img.w + xa]; val p11 = img.px[(ya + 1) * img.w + xa + 1]
        var out = 0
        for (sh in intArrayOf(16, 8, 0)) {
            val v = (((p00 shr sh) and 255) * (1 - fx) + ((p10 shr sh) and 255) * fx) * (1 - fy) +
                    (((p01 shr sh) and 255) * (1 - fx) + ((p11 shr sh) and 255) * fx) * fy
            out = (out shl 8) or (v + .5).toInt().coerceIn(0, 255)
        }
        return out
    }

    /** 3x3 homography (h33 = 1, row-major 8 values) taking src[i] to dst[i]: cv2.getPerspectiveTransform. */
    fun homography(src: List<P>, dst: List<P>): DoubleArray {
        val a = Array(8) { DoubleArray(9) }
        for (i in 0..3) {
            val (u, v) = src[i]
            val (x, y) = dst[i]
            a[2 * i] = doubleArrayOf(u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x, x)
            a[2 * i + 1] = doubleArrayOf(0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y, y)
        }
        for (c in 0..7) {  // Gaussian elimination with partial pivoting
            val p = (c..7).maxBy { abs(a[it][c]) }
            val t = a[c]; a[c] = a[p]; a[p] = t
            for (r in 0..7) if (r != c) {
                val g = a[r][c] / a[c][c]
                for (q in c..8) a[r][q] -= g * a[c][q]
            }
        }
        return DoubleArray(8) { a[it][8] / a[it][it] }
    }

    private fun dist(a: P, b: P) = hypot(a.x - b.x, a.y - b.y)

    private fun byAngle(pts: List<P>): List<P> {
        val mx = pts.sumOf { it.x } / pts.size
        val my = pts.sumOf { it.y } / pts.size
        return pts.sortedBy { atan2(it.y - my, it.x - mx) }
    }

    /** A photo of the pond's square, however tilted, keeps every corner within 50..130 degrees and opposite sides within
     *  a factor 1.8; a quad that picked up a lily pad's centre instead of a corner doesn't (koi.py _squareish). */
    private fun squareish(q: List<P>): Boolean {
        for (i in 0..3) {
            val a = q[(i + 3) % 4]; val b = q[i]; val c = q[(i + 1) % 4]
            val ux = a.x - b.x; val uy = a.y - b.y; val vx = c.x - b.x; val vy = c.y - b.y
            val cos = (ux * vx + uy * vy) / (hypot(ux, uy) * hypot(vx, vy))
            if (Math.toDegrees(kotlin.math.acos(cos.coerceIn(-1.0, 1.0))) !in 50.0..130.0) return false
        }
        val s = (0..3).map { dist(q[it], q[(it + 1) % 4]) }
        return max(s[0], s[2]) < 1.8 * min(s[0], s[2]) && max(s[1], s[3]) < 1.8 * min(s[1], s[3])
    }

    /** Area of a quadrilateral given in angular order, or null if it isn't convex. */
    private fun convexArea(q: List<P>): Double? {
        var sign = 0
        var area = 0.0
        for (i in 0..3) {
            val a = q[i]; val b = q[(i + 1) % 4]; val c = q[(i + 2) % 4]
            val cr = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            val s = if (cr > 0) 1 else if (cr < 0) -1 else 0
            if (s == 0 || (sign != 0 && s != sign)) return null
            sign = s
            area += a.x * b.y - b.x * a.y
        }
        return abs(area) / 2
    }

    private fun combos(n: Int, k: Int): List<IntArray> {
        val out = ArrayList<IntArray>()
        fun rec(start: Int, acc: IntArray, d: Int) {
            if (d == k) { out += acc.copyOf(); return }
            for (i in start until n) { acc[d] = i; rec(i + 1, acc, d + 1) }
        }
        rec(0, IntArray(k), 0)
        return out
    }
}
