package app.koipond.core

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Packed 0xRRGGBB image. */
class Rgb(val w: Int, val h: Int, val px: IntArray = IntArray(w * h))

/** Single-channel float image (pixel (x, y) has its centre at integer coordinates, as in OpenCV). */
class FImg(val w: Int, val h: Int, val a: FloatArray = FloatArray(w * h)) {
    operator fun get(x: Int, y: Int) = a[y * w + x]
}

// --- ink: anti-aliased strokes as coverage, unioned by max (overlaps never get darker) ---------

class Ink(val w: Int, val h: Int) {
    val img = FImg(w, h)
    private val a = img.a

    fun seg(x0: Double, y0: Double, x1: Double, y1: Double, width: Double, alpha: Double) {
        val r = width / 2
        val xa = max(0, floor(min(x0, x1) - r - 1).toInt())
        val xb = min(w - 1, ceil(max(x0, x1) + r + 1).toInt())
        val ya = max(0, floor(min(y0, y1) - r - 1).toInt())
        val yb = min(h - 1, ceil(max(y0, y1) + r + 1).toInt())
        val dx = x1 - x0
        val dy = y1 - y0
        val l2 = dx * dx + dy * dy
        for (py in ya..yb) for (px in xa..xb) {
            var t = if (l2 > 0) ((px - x0) * dx + (py - y0) * dy) / l2 else 0.0
            t = t.coerceIn(0.0, 1.0)
            val ex = px - (x0 + t * dx)
            val ey = py - (y0 + t * dy)
            val d = sqrt(ex * ex + ey * ey)
            val c = (r + .5 - d).coerceIn(0.0, 1.0) * alpha
            val i = py * w + px
            if (c > a[i]) a[i] = c.toFloat()
        }
    }

    fun line(p: Pts, width: Double, alpha: Double = 1.0) {
        if (p.n == 1) disc(p[0], width / 2, alpha)
        for (i in 0 until p.n - 1) seg(p.x[i], p.y[i], p.x[i + 1], p.y[i + 1], width, alpha)
    }

    private inline fun radial(c: P, reach: Double, f: (Double, Int) -> Unit) {
        val xa = max(0, floor(c.x - reach - 1).toInt())
        val xb = min(w - 1, ceil(c.x + reach + 1).toInt())
        val ya = max(0, floor(c.y - reach - 1).toInt())
        val yb = min(h - 1, ceil(c.y + reach + 1).toInt())
        for (py in ya..yb) for (px in xa..xb) { val ex = px - c.x; val ey = py - c.y; f(sqrt(ex * ex + ey * ey), py * w + px) }
    }

    fun disc(c: P, r: Double, alpha: Double = 1.0) = radial(c, r) { d, i ->
        val v = ((r + .5 - d).coerceIn(0.0, 1.0) * alpha).toFloat()
        if (v > a[i]) a[i] = v
    }

    /** Outline ring like PIL's ellipse(outline, width): the band of width wd just inside radius r. */
    fun ring(c: P, r: Double, wd: Double, alpha: Double = 1.0) = radial(c, r) { d, i ->
        val v = ((wd / 2 + .5 - abs(d - (r - wd / 2))).coerceIn(0.0, 1.0) * alpha).toFloat()
        if (v > a[i]) a[i] = v
    }

    fun erase(c: P, r: Double) = radial(c, r) { d, i -> a[i] *= (1 - (r + .5 - d).coerceIn(0.0, 1.0)).toFloat() }
}

// --- byte masks with polygon fill (cv2.fillPoly: pixel centres inside the polygon) -----------

class Mask(val w: Int, val h: Int, val a: ByteArray = ByteArray(w * h)) {
    fun fillPoly(p: Pts, value: Double = 1.0, scale: Double = 1.0) {
        val v = (value * 255).roundToInt().coerceIn(0, 255).toByte()
        val n = p.n
        var y0 = Double.MAX_VALUE
        var y1 = -Double.MAX_VALUE
        for (i in 0 until n) { y0 = min(y0, p.y[i] * scale); y1 = max(y1, p.y[i] * scale) }
        val xs = DoubleArray(n)
        for (y in max(0, ceil(y0).toInt())..min(h - 1, floor(y1).toInt())) {
            var k = 0
            for (i in 0 until n) {
                val j = (i + 1) % n
                val ya = p.y[i] * scale
                val yb = p.y[j] * scale
                if ((ya <= y && y < yb) || (yb <= y && y < ya)) xs[k++] = p.x[i] * scale + (y - ya) / (yb - ya) * (p.x[j] - p.x[i]) * scale
            }
            xs.sort(0, k)
            var q = 0
            while (q + 1 < k) {
                for (x in max(0, ceil(xs[q]).toInt())..min(w - 1, floor(xs[q + 1]).toInt())) a[y * w + x] = v
                q += 2
            }
        }
    }

    fun fillCircle(cx: Double, cy: Double, r: Double) {
        for (y in max(0, floor(cy - r).toInt())..min(h - 1, ceil(cy + r).toInt()))
            for (x in max(0, floor(cx - r).toInt())..min(w - 1, ceil(cx + r).toInt()))
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) a[y * w + x] = -1
    }

    fun toFloat() = FImg(w, h, FloatArray(w * h) { (a[it].toInt() and 0xff) / 255f })
    operator fun get(x: Int, y: Int) = a[y * w + x].toInt() != 0
}

// --- filters -----------------------------------------------------------------------------

private fun reflect101(i: Int, n: Int): Int {
    if (n == 1) return 0
    var j = i
    while (j < 0 || j >= n) j = if (j < 0) -j else 2 * n - 2 - j
    return j
}

/** Gaussian blur, 4-sigma kernel, reflect-101 borders (cv2.GaussianBlur on float images). */
fun gaussBlur(src: FImg, sigma: Double): FImg {
    val r = max(1, ceil(4 * sigma).toInt())
    val k = FloatArray(2 * r + 1) { exp(-((it - r) * (it - r)) / (2 * sigma * sigma)).toFloat() }
    val s = k.sum()
    for (i in k.indices) k[i] /= s
    val w = src.w
    val h = src.h
    val tmp = FloatArray(w * h)
    for (y in 0 until h) {
        val row = y * w
        for (x in 0 until w) {
            var acc = 0f
            if (x >= r && x < w - r) for (t in -r..r) acc += k[t + r] * src.a[row + x + t]
            else for (t in -r..r) acc += k[t + r] * src.a[row + reflect101(x + t, w)]
            tmp[row + x] = acc
        }
    }
    val out = FloatArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        var acc = 0f
        if (y >= r && y < h - r) for (t in -r..r) acc += k[t + r] * tmp[(y + t) * w + x]
        else for (t in -r..r) acc += k[t + r] * tmp[reflect101(y + t, h) * w + x]
        out[y * w + x] = acc
    }
    return FImg(w, h, out)
}

/** 1-D sliding max/min over a centred window of size k, ignoring samples past the ends. */
private fun slide(src: FloatArray, off: Int, stride: Int, n: Int, k: Int, isMax: Boolean, dst: FloatArray) {
    val half = k / 2
    val dq = IntArray(n)
    var head = 0
    var tail = 0
    var next = 0
    for (i in 0 until n) {
        val hi = min(n - 1, i + half)
        while (next <= hi) {
            val v = src[off + next * stride]
            while (tail > head && (if (isMax) src[off + dq[tail - 1] * stride] <= v else src[off + dq[tail - 1] * stride] >= v)) tail--
            dq[tail++] = next
            next++
        }
        while (dq[head] < i - (k - 1 - half)) head++
        dst[off + i * stride] = src[off + dq[head] * stride]
    }
}

fun rectFilter(src: FImg, k: Int, isMax: Boolean): FImg {
    val tmp = FloatArray(src.a.size)
    for (y in 0 until src.h) slide(src.a, y * src.w, 1, src.w, k, isMax, tmp)
    val out = FloatArray(src.a.size)
    for (x in 0 until src.w) slide(tmp, x, src.w, src.h, k, isMax, out)
    return FImg(src.w, src.h, out)
}

fun closeRect(src: FImg, k: Int) = rectFilter(rectFilter(src, k, true), k, false)

/** Kernel offsets as primitive arrays, so hot loops never box. */
class Offs(val dx: IntArray, val dy: IntArray) { val n get() = dx.size }

/** OpenCV's MORPH_ELLIPSE structuring element as (dx, dy) offsets. */
fun ellipseOffsets(n: Int): Offs {
    val r = n / 2
    val c = n / 2
    val out = ArrayList<Pair<Int, Int>>()
    for (i in 0 until n) {
        val dy = i - r
        if (abs(dy) > r) continue
        val dx = if (r == 0) 0 else (c * sqrt((r * r - dy * dy).toDouble() / (r * r))).roundToInt()
        for (j in max(c - dx, 0) until min(c + dx + 1, n)) out += (j - c) to dy
    }
    return Offs(IntArray(out.size) { out[it].first }, IntArray(out.size) { out[it].second })
}

/** Grayscale morphology with a small explicit kernel (cv2.morphologyEx OPEN = erode then dilate). */
fun openGray(src: FImg, offs: Offs): FImg {
    fun pass(a: FloatArray, isMax: Boolean): FloatArray {
        val out = FloatArray(a.size)
        for (y in 0 until src.h) for (x in 0 until src.w) {
            var v = if (isMax) -Float.MAX_VALUE else Float.MAX_VALUE
            for (o in 0 until offs.n) {
                val xx = x + offs.dx[o]
                val yy = y + offs.dy[o]
                if (xx < 0 || yy < 0 || xx >= src.w || yy >= src.h) continue
                val s = a[yy * src.w + xx]
                v = if (isMax) max(v, s) else min(v, s)
            }
            out[y * src.w + x] = v
        }
        return out
    }
    return FImg(src.w, src.h, pass(pass(src.a, false), true))
}

/** Exact Euclidean distance from every nonzero pixel to the nearest zero pixel (Felzenszwalb-Huttenlocher). */
fun edt(mask: ByteArray, w: Int, h: Int): FImg {
    val inf = 1e20
    val f = DoubleArray(w * h) { if (mask[it].toInt() != 0) inf else 0.0 }
    val n = max(w, h)
    val v = IntArray(n)
    val z = DoubleArray(n + 1)
    val d = DoubleArray(n)
    val g = DoubleArray(n)
    fun pass1d(off: Int, stride: Int, len: Int) {
        for (q in 0 until len) g[q] = f[off + q * stride]
        var k = 0
        v[0] = 0; z[0] = -inf; z[1] = inf
        for (q in 1 until len) {
            var s: Double
            while (true) {
                s = ((g[q] + q.toDouble() * q) - (g[v[k]] + v[k].toDouble() * v[k])) / (2.0 * q - 2.0 * v[k])
                if (s <= z[k] && k > 0) k-- else break
            }
            if (s <= z[k]) { v[k] = q; z[k] = -inf; z[k + 1] = inf; continue }
            k++
            v[k] = q; z[k] = s; z[k + 1] = inf
        }
        k = 0
        for (q in 0 until len) {
            while (z[k + 1] < q) k++
            d[q] = (q - v[k]).toDouble() * (q - v[k]) + g[v[k]]
        }
        for (q in 0 until len) f[off + q * stride] = d[q]
    }
    for (x in 0 until w) pass1d(x, w, h)
    for (y in 0 until h) pass1d(y * w, 1, w)
    return FImg(w, h, FloatArray(w * h) { sqrt(f[it]).toFloat() })
}

/** 4-connected components of nonzero pixels: labels (0 = background) and count+1. */
fun components(mask: ByteArray, w: Int, h: Int): Pair<IntArray, Int> {
    val lab = IntArray(w * h)
    val q = IntArray(w * h)
    var n = 1
    for (s in 0 until w * h) {
        if (mask[s].toInt() == 0 || lab[s] != 0) continue
        var head = 0
        var tail = 0
        q[tail++] = s
        lab[s] = n
        while (head < tail) {
            val p = q[head++]
            val x = p % w
            val y = p / w
            if (x > 0 && mask[p - 1].toInt() != 0 && lab[p - 1] == 0) { lab[p - 1] = n; q[tail++] = p - 1 }
            if (x < w - 1 && mask[p + 1].toInt() != 0 && lab[p + 1] == 0) { lab[p + 1] = n; q[tail++] = p + 1 }
            if (y > 0 && mask[p - w].toInt() != 0 && lab[p - w] == 0) { lab[p - w] = n; q[tail++] = p - w }
            if (y < h - 1 && mask[p + w].toInt() != 0 && lab[p + w] == 0) { lab[p + w] = n; q[tail++] = p + w }
        }
        n++
    }
    return lab to n
}

/** Percentile of values in [0, hi] via a fine histogram (exact enough for thresholds). */
fun percentile(a: FloatArray, q: Double, hi: Float, keep: (Float) -> Boolean = { true }): Float {
    val bins = 8192
    val hist = IntArray(bins)
    var total = 0
    for (v in a) if (keep(v)) { hist[((v / hi) * (bins - 1)).toInt().coerceIn(0, bins - 1)]++; total++ }
    if (total == 0) return 0f
    val target = q / 100 * total
    var acc = 0.0
    for (b in 0 until bins) { acc += hist[b]; if (acc >= target) return (b + .5f) / (bins - 1) * hi }
    return hi
}

/** cv2.remap INTER_LINEAR with a zero border. */
fun sampleBilinear(img: FImg, x: Double, y: Double): Float {
    val x0 = floor(x).toInt()
    val y0 = floor(y).toInt()
    val fx = (x - x0).toFloat()
    val fy = (y - y0).toFloat()
    fun at(xx: Int, yy: Int) = if (xx < 0 || yy < 0 || xx >= img.w || yy >= img.h) 0f else img.a[yy * img.w + xx]
    return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy
}

/** scipy median_filter, 1-D, mode 'nearest'. */
fun median1d(a: DoubleArray, size: Int): DoubleArray {
    val h = size / 2
    val win = DoubleArray(size)
    return DoubleArray(a.size) { i ->
        for (t in 0 until size) win[t] = a[(i + t - h).coerceIn(0, a.size - 1)]
        win.sort()
        win[h]
    }
}

/** scipy gaussian_filter1d, mode 'reflect' (half-sample symmetric), truncate 4. */
fun gauss1d(a: DoubleArray, sigma: Double): DoubleArray {
    val r = (4 * sigma + .5).toInt()
    val k = DoubleArray(2 * r + 1) { exp(-((it - r) * (it - r)) / (2 * sigma * sigma)) }
    val s = k.sum()
    fun refl(i: Int): Int {
        var j = i
        val n = a.size
        while (j < 0 || j >= n) j = if (j < 0) -j - 1 else 2 * n - 1 - j
        return j
    }
    return DoubleArray(a.size) { i -> var acc = 0.0; for (t in -r..r) acc += k[t + r] * a[refl(i + t)]; acc / s }
}

/** 2x2 area downsample (odd edges averaged over what exists). */
fun down2(src: FImg): FImg {
    val w2 = (src.w + 1) / 2
    val h2 = (src.h + 1) / 2
    val out = FloatArray(w2 * h2)
    for (y in 0 until h2) for (x in 0 until w2) {
        var s = 0f
        var c = 0
        for (dy in 0..1) for (dx in 0..1) {
            val xx = 2 * x + dx
            val yy = 2 * y + dy
            if (xx < src.w && yy < src.h) { s += src.a[yy * src.w + xx]; c++ }
        }
        out[y * w2 + x] = s / c
    }
    return FImg(w2, h2, out)
}

/** Bilinear upsample of a half-resolution image back to (w, h). */
fun up2(src: FImg, w: Int, h: Int): FImg {
    val out = FloatArray(w * h)
    for (y in 0 until h) {
        val sy = ((y + .5f) / 2 - .5f).coerceIn(0f, (src.h - 1).toFloat())
        val y0 = sy.toInt()
        val y1 = min(y0 + 1, src.h - 1)
        val fy = sy - y0
        for (x in 0 until w) {
            val sx = ((x + .5f) / 2 - .5f).coerceIn(0f, (src.w - 1).toFloat())
            val x0 = sx.toInt()
            val x1 = min(x0 + 1, src.w - 1)
            val fx = sx - x0
            val top = src.a[y0 * src.w + x0] * (1 - fx) + src.a[y0 * src.w + x1] * fx
            val bot = src.a[y1 * src.w + x0] * (1 - fx) + src.a[y1 * src.w + x1] * fx
            out[y * w + x] = top * (1 - fy) + bot * fy
        }
    }
    return FImg(w, h, out)
}

private fun cubicWeights(n: Int, size: Int): Pair<IntArray, FloatArray> {
    val a = -.75
    fun k(x: Double): Double {
        val ax = abs(x)
        return if (ax <= 1) (a + 2) * ax * ax * ax - (a + 3) * ax * ax + 1
        else if (ax < 2) a * ax * ax * ax - 5 * a * ax * ax + 8 * a * ax - 4 * a else 0.0
    }
    val idx = IntArray(size * 4)
    val wts = FloatArray(size * 4)
    for (o in 0 until size) {
        val s = (o + .5) * n / size - .5
        val i0 = floor(s).toInt()
        val t = s - i0
        val kw = doubleArrayOf(k(1 + t), k(t), k(1 - t), k(2 - t))
        for (j in 0..3) { idx[o * 4 + j] = (i0 - 1 + j).coerceIn(0, n - 1); wts[o * 4 + j] = kw[j].toFloat() }
    }
    return idx to wts
}

/** cv2.resize INTER_CUBIC from a tiny n x n grid up to (w, h): smooth low-frequency noise (separable). */
fun upscaleCubic(src: DoubleArray, n: Int, w: Int, h: Int): FImg {
    val (xi, xw) = cubicWeights(n, w)
    val (yi, yw) = cubicWeights(n, h)
    val rows = FloatArray(n * w)  // each source row stretched to width w
    for (r in 0 until n) for (x in 0 until w) {
        var acc = 0f
        for (j in 0..3) acc += xw[x * 4 + j] * src[r * n + xi[x * 4 + j]].toFloat()
        rows[r * w + x] = acc
    }
    val out = FloatArray(w * h)
    for (y in 0 until h) for (x in 0 until w) {
        var acc = 0f
        for (j in 0..3) acc += yw[y * 4 + j] * rows[yi[y * 4 + j] * w + x]
        out[y * w + x] = acc
    }
    return FImg(w, h, out)
}
