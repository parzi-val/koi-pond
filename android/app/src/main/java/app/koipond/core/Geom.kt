package app.koipond.core

import java.util.SplittableRandom
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/** A polyline: parallel x / y arrays (the Kotlin stand-in for numpy (n, 2) point arrays). */
class Pts(val x: DoubleArray, val y: DoubleArray) {
    val n get() = x.size
    operator fun get(i: Int) = P(x[i], y[i])
    fun reversed() = Pts(x.reversedArray(), y.reversedArray())
    fun sub(a: Int, b: Int) = Pts(x.copyOfRange(a, b), y.copyOfRange(a, b))
    operator fun plus(o: Pts) = Pts(x + o.x, y + o.y)
    fun closed() = Pts(x + x[0], y + y[0])

    companion object {
        fun of(n: Int, f: (Int) -> P): Pts {
            val xs = DoubleArray(n)
            val ys = DoubleArray(n)
            for (i in 0 until n) {
                val p = f(i)
                xs[i] = p.x; ys[i] = p.y
            }
            return Pts(xs, ys)
        }
        fun of(vararg ps: P) = of(ps.size) { ps[it] }
    }
}

data class P(val x: Double, val y: Double) {
    operator fun plus(o: P) = P(x + o.x, y + o.y)
    operator fun minus(o: P) = P(x - o.x, y - o.y)
    operator fun times(s: Double) = P(x * s, y * s)
    fun norm() = hypot(x, y)
    fun unit(): P { val n = norm(); return if (n == 0.0) this else P(x / n, y / n) }
}

fun linspace(a: Double, b: Double, n: Int) = DoubleArray(n) { if (n == 1) a else a + (b - a) * it / (n - 1) }

/** numpy.interp: piecewise linear, clamped at the ends; xp increasing. */
fun interp(x: Double, xp: DoubleArray, fp: DoubleArray): Double {
    if (x <= xp[0]) return fp[0]
    if (x >= xp[xp.size - 1]) return fp[fp.size - 1]
    var lo = 0
    var hi = xp.size - 1
    while (hi - lo > 1) {
        val m = (lo + hi) ushr 1
        if (xp[m] <= x) lo = m else hi = m
    }
    val t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] + t * (fp[hi] - fp[lo])
}

fun cumlen(p: Pts): DoubleArray {
    val s = DoubleArray(p.n)
    for (i in 1 until p.n) s[i] = s[i - 1] + hypot(p.x[i] - p.x[i - 1], p.y[i] - p.y[i - 1])
    return s
}

/** numpy.gradient along a polyline (central inside, one-sided at the ends). */
fun gradient(p: Pts): Pts = Pts.of(p.n) { i ->
    when {
        p.n < 2 -> P(0.0, 0.0)
        i == 0 -> P(p.x[1] - p.x[0], p.y[1] - p.y[0])
        i == p.n - 1 -> P(p.x[i] - p.x[i - 1], p.y[i] - p.y[i - 1])
        else -> P((p.x[i + 1] - p.x[i - 1]) / 2, (p.y[i + 1] - p.y[i - 1]) / 2)
    }
}

fun resampleN(p: Pts, n: Int): Pts {
    val s = cumlen(p)
    val t = linspace(0.0, s.last(), n)
    return Pts(DoubleArray(n) { interp(t[it], s, p.x) }, DoubleArray(n) { interp(t[it], s, p.y) })
}

fun resampleStep(p: Pts, step: Double = 2.0): Pts {
    val len = cumlen(p).last()
    return resampleN(p, maxOf(2, (len / step).toInt() + 1))
}

/** Hand-drawn wobble: push a polyline along its normal by smooth noise (amp px, wavelength wl px). */
fun wobble(p0: Pts, amp: Double, wl: Double, rng: Rng): Pts {
    val p = resampleStep(p0)
    if (p.n < 3 || amp == 0.0) return p
    val s = cumlen(p)
    val g = gradient(p)
    val a = rng.uniform(0.0, 2 * PI)
    val b = rng.uniform(0.0, 2 * PI)
    return Pts.of(p.n) { i ->
        val nn = hypot(g.x[i], g.y[i]) + 1e-9
        val off = amp * (.7 * sin(2 * PI * s[i] / wl + a) + .3 * sin(2 * PI * s[i] / (.37 * wl) + b))
        P(p.x[i] - g.y[i] / nn * off, p.y[i] + g.x[i] / nn * off)
    }
}

/** Fin membrane margin: soft scallops between ray tips, dipping toward the base by `dip`. */
fun smoothMargin(tips: Pts, base: Pts, dip: Double): Pts {
    val xs = ArrayList<Double>()
    val ys = ArrayList<Double>()
    xs += tips.x[0]; ys += tips.y[0]
    for (i in 0 until tips.n - 1) {
        val cx = (tips.x[i] + tips.x[i + 1]) / 2 * (1 - dip) + (base.x[i] + base.x[i + 1]) / 2 * dip
        val cy = (tips.y[i] + tips.y[i + 1]) / 2 * (1 - dip) + (base.y[i] + base.y[i + 1]) / 2 * dip
        for (k in 1..5) {
            val q = k / 5.0
            xs += (1 - q) * (1 - q) * tips.x[i] + 2 * q * (1 - q) * cx + q * q * tips.x[i + 1]
            ys += (1 - q) * (1 - q) * tips.y[i] + 2 * q * (1 - q) * cy + q * q * tips.y[i + 1]
        }
    }
    return Pts(xs.toDoubleArray(), ys.toDoubleArray())
}

/** Deterministic randomness for a pond style (same seed -> same pond, on every device). */
class Rng(seed: Long) {
    private val r = SplittableRandom(seed)
    private var spare = Double.NaN
    fun uniform() = r.nextDouble()
    fun uniform(a: Double, b: Double) = a + (b - a) * r.nextDouble()
    fun int(lo: Int, hi: Int) = lo + r.nextInt(hi - lo)  // [lo, hi), like numpy integers
    fun normal(sd: Double = 1.0): Double {
        if (!spare.isNaN()) { val s = spare; spare = Double.NaN; return s * sd }
        var u: Double
        var v: Double
        var q: Double
        do { u = 2 * r.nextDouble() - 1; v = 2 * r.nextDouble() - 1; q = u * u + v * v } while (q >= 1 || q == 0.0)
        val f = sqrt(-2 * ln(q) / q)
        spare = v * f
        return u * f * sd
    }
    fun <T> choice(xs: List<T>) = xs[r.nextInt(xs.size)]
    fun permutation(n: Int): IntArray {
        val a = IntArray(n) { it }
        for (i in n - 1 downTo 1) { val j = r.nextInt(i + 1); val t = a[i]; a[i] = a[j]; a[j] = t }
        return a
    }
}

fun polar(c: P, r: Double, a: Double) = P(c.x + r * cos(a), c.y + r * sin(a))

/** Tiny section profiler (wall time per named stage), read out by the JVM tests. */
object Prof {
    val ms = LinkedHashMap<String, Long>()
    inline fun <T> sec(name: String, block: () -> T): T {
        val t = System.nanoTime()
        try { return block() } finally { ms[name] = (ms[name] ?: 0L) + (System.nanoTime() - t) / 1_000_000 }
    }
    fun report() = ms.entries.joinToString("  ") { "${it.key} ${it.value}ms" }.also { ms.clear() }
}
