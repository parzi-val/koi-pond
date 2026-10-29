package app.koipond.core

import kotlin.math.exp
import kotlin.math.roundToInt

/**
 * The pond's per-pixel colour passes, each a small function: JIT compilers (HotSpot, ART)
 * skip huge methods, so loops left inside the big render function would run interpreted.
 */
internal object Wash {
    fun paper(n: Int): Array<FloatArray> = Array(3) { c -> FloatArray(n).also { it.fill(KoiSpec.PAPER[c].toFloat()) } }

    fun water(img: Array<FloatArray>, low: FloatArray, col: DoubleArray) {
        for (c in 0..2) {
            val wc = (1 - col[c]).toFloat()
            val ch = img[c]
            for (i in ch.indices) ch[i] *= 1 - (.8f + low[i]) * wc
        }
    }

    /** Koi bodies are paper, with a wide, gentle darkening toward the flanks. */
    fun bodies(img: Array<FloatArray>, bodies: ByteArray, edge: FloatArray) {
        for (i in bodies.indices) if (bodies[i].toInt() != 0) {
            val sh = (1 - .07 * exp(-edge[i] / 45.0)).toFloat()
            for (c in 0..2) img[c][i] = KoiSpec.PAPER[c].toFloat() * sh
        }
    }

    /** A soft-edged wash mask: blurred at half resolution (it is soft anyway), back at full size. Null if empty. */
    fun soft(m: Mask, sigma: Double): FImg? {
        if (m.a.all { it.toInt() == 0 }) return null
        return up2(gaussBlur(down2(m.toFloat()), sigma / 2), m.w, m.h)
    }

    fun tint(img: Array<FloatArray>, a: FloatArray, col: DoubleArray, gain: Float, clip: FloatArray?) {
        for (c in 0..2) {
            val k = (1 - col[c]).toFloat()
            val ch = img[c]
            if (clip == null) for (i in ch.indices) { val ai = a[i] * gain; if (ai > 0) ch[i] *= 1 - ai * k }
            else for (i in ch.indices) { val ai = a[i] * gain * clip[i]; if (ai > 0) ch[i] *= 1 - ai * k }
        }
    }

    /** Translucent fins: lighten the water toward paper (never over a body). */
    fun lighten(img: Array<FloatArray>, a: FloatArray, bodies: ByteArray) {
        for (i in a.indices) if (bodies[i].toInt() == 0 && a[i] > 0)
            for (c in 0..2) img[c][i] += (KoiSpec.PAPER[c].toFloat() - img[c][i]) * a[i]
    }

    /** Petals sit on the pad, not tinted by it. */
    fun cut(green: ByteArray, pink: ByteArray) {
        for (i in green.indices) {
            val pk = pink[i].toInt() and 0xff
            if (pk > 0) green[i] = ((green[i].toInt() and 0xff) * (1 - pk / 255.0)).roundToInt().toByte()
        }
    }

    fun composite(img: Array<FloatArray>, ink: FloatArray, out: IntArray) {
        val ir = KoiSpec.INK[0].toFloat()
        val ig = KoiSpec.INK[1].toFloat()
        val ib = KoiSpec.INK[2].toFloat()
        for (i in out.indices) {
            val a = ink[i]
            val r = (img[0][i] * (1 - a) + ir * a + .5f).toInt().coerceIn(0, 255)
            val g = (img[1][i] * (1 - a) + ig * a + .5f).toInt().coerceIn(0, 255)
            val b = (img[2][i] * (1 - a) + ib * a + .5f).toInt().coerceIn(0, 255)
            out[i] = (r shl 16) or (g shl 8) or b
        }
    }
}
