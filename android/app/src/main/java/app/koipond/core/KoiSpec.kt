package app.koipond.core

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin

/**
 * The koi-pond format, mirrored from koi.py: the pond geometry ranges, the scale grid every
 * koi carries (cells in bit order), the templates the decoder reads with, and the payload
 * layout. Changing anything here makes ponds unreadable by the other side.
 */
object KoiSpec {
    const val S = 2400
    const val NF = 6
    const val BOLD = 3.6
    const val THIN = 1.3
    val L_RNG = 640.0 to 700.0
    val HW_RNG = 90.0 to 110.0
    val P_RNG = 16.0 to 22.0
    val C_RNG = .6 to .75
    val D_RNG = .6 to .85
    val K_RNG = .02 to .04
    // corner anchors (koi.py ANCHORS): a bold ring + centre dot in each corner; the decoder unwarps the pond from them
    const val ANCHOR_M = 125.0
    const val ANCHOR_R = 80.0
    const val ANCHOR_W = 20.0
    const val ANCHOR_DOT = 26.0
    val ANCHORS = listOf(P(ANCHOR_M, ANCHOR_M), P(S - ANCHOR_M, ANCHOR_M), P(S - ANCHOR_M, S - ANCHOR_M), P(ANCHOR_M, S - ANCHOR_M))
    const val RECT = 1600  // side of the unwarped pond the decoder reads
    const val COLS = 24
    const val U0 = .22
    const val U1 = .88
    const val VM = .86
    const val DV_PX = 15.0
    val PAPER = doubleArrayOf(255.0, 250.0, 240.0)
    val INK = doubleArrayOf(24.0, 20.0, 18.0)

    /** Half-width of a koi seen from above: broad shoulders, rounded snout (k), tail stalk P. */
    fun profile(u0: Double, hw: Double, c: Double, d: Double, p: Double, k: Double): Double {
        val u = u0.coerceIn(0.0, 1.0)
        return hw * sin(PI * u.pow(c)).pow(d) * (1 - exp(-u / k)) + p * u * u
    }

    private fun minHalf(u: Double): Double {
        val g = linspace(0.0, 1.0, 5)
        var best = Double.MAX_VALUE
        for (gd in g) for (gc in g) {
            val c = C_RNG.first + gc * (C_RNG.second - C_RNG.first)
            val d = D_RNG.first + gd * (D_RNG.second - D_RNG.first)
            best = minOf(best, profile(u, HW_RNG.first, c, d, P_RNG.first, K_RNG.second))
        }
        return best
    }

    class Cell(val u: Double, val v: Double, val du: Double, val dv: Double, val col: Int)

    /** Odd columns have one row fewer, offset half a row, so a mirrored koi lands on the same grid. */
    private fun cells(cols: IntRange): List<Cell> {
        val du = (U1 - U0) / COLS
        val out = ArrayList<Cell>()
        for (c in cols) {
            val u = U0 + (c + .5) * du
            val n0 = maxOf(2, (2 * VM * minHalf(u) / DV_PX).toInt())
            val dv = 2 * VM / n0
            val n = n0 - c % 2
            for (j in 0 until n) out += Cell(u, (j - (n - 1) / 2.0) * dv, du, dv, c)
        }
        return out
    }

    val CELLS = cells(0 until COLS)
    val EXTRA = cells(COLS until COLS + 2)              // decorative scales toward the tail, not read
    val FISH_BITS = CELLS.size / 8 * 8
    val CHUNK = FISH_BITS / 8 - 5                        // per koi: index byte + chunk + 4 inner parity
    val PAYLOAD_BYTES = (NF - 1) * CHUNK                 // outer code survives losing any one koi
    val PARITY_BYTES = CHUNK
    const val INNER_NSYM = 4
    val COL = IntArray(CELLS.size) { CELLS[it].col }
    val STARTS = IntArray(COLS + 1) { c -> CELLS.indexOfFirst { it.col >= c }.let { if (it < 0) CELLS.size else it } }
    val MIRROR = IntArray(CELLS.size) { k ->
        val c = COL[k]
        val a = STARTS[c]
        val n = STARTS[c + 1] - a
        a + (n - 1 - (k - a))
    }

    // templates, in cell units relative to the scale centre: per curve, du offsets and dv offsets
    private val HT = DoubleArray(6) { Math.toRadians(40.0 + 8.0 * it) }
    private val RR = doubleArrayOf(.45, .6, .75)
    val HATCH_U: DoubleArray = DoubleArray(18) { -.55 + RR[it / 6] * cos(HT[it % 6]) }
    val HATCH_V: DoubleArray = DoubleArray(18) { .5 * RR[it / 6] * sin(HT[it % 6]) }  // dorsal curve: -V, ventral: +V
    val ARC_U: DoubleArray = DoubleArray(9) { -.55 + cos(Math.toRadians(-80.0 + 20.0 * it)) }
    val ARC_V: DoubleArray = DoubleArray(9) { .5 * sin(Math.toRadians(-80.0 + 20.0 * it)) }

    /** Local tracking search per column: shift along u, slant, shift along v, stretch along v. */
    val COMBOS: Array<DoubleArray> = run {
        val axes = listOf(linspace(-.35, .35, 7), linspace(-.3, .3, 5), linspace(-.25, .25, 5), linspace(-.3, .3, 5))
        val out = ArrayList<DoubleArray>()
        for (a in axes[0]) for (b in axes[1]) for (c in axes[2]) for (d in axes[3]) out += doubleArrayOf(a, b, c, d)
        out.toTypedArray()
    }
}
