package app.koipond.core

/**
 * Reed-Solomon over GF(2^8), byte-compatible with Python's `reedsolo` defaults
 * (prim 0x11d, generator 2, fcr 0, systematic: message bytes then nsym parity bytes),
 * so ponds drawn by the Python reference and by this app read on either side.
 * Decodes errors and erasures (Berlekamp-Massey + Forney), as in reedsolo.
 */
class RsException(msg: String) : Exception(msg)

object Rs {
    private val exp = IntArray(512)
    private val log = IntArray(256)

    init {
        var x = 1
        for (i in 0 until 255) {
            exp[i] = x
            log[x] = i
            x = x shl 1
            if (x and 0x100 != 0) x = x xor 0x11d
        }
        for (i in 255 until 512) exp[i] = exp[i - 255]
    }

    private fun mul(a: Int, b: Int) = if (a == 0 || b == 0) 0 else exp[log[a] + log[b]]
    private fun div(a: Int, b: Int): Int {
        if (b == 0) throw RsException("division by zero")
        return if (a == 0) 0 else exp[(log[a] + 255 - log[b]) % 255]
    }
    private fun pow(x: Int, p: Int) = exp[Math.floorMod(log[x] * p, 255)]
    private fun inv(x: Int) = exp[255 - log[x]]

    private fun polyScale(p: IntArray, x: Int) = IntArray(p.size) { mul(p[it], x) }
    private fun polyAdd(p: IntArray, q: IntArray): IntArray {
        val r = IntArray(maxOf(p.size, q.size))
        for (i in p.indices) r[i + r.size - p.size] = p[i]
        for (i in q.indices) r[i + r.size - q.size] = r[i + r.size - q.size] xor q[i]
        return r
    }
    private fun polyMul(p: IntArray, q: IntArray): IntArray {
        val r = IntArray(p.size + q.size - 1)
        for (j in q.indices) for (i in p.indices) r[i + j] = r[i + j] xor mul(p[i], q[j])
        return r
    }
    private fun polyEval(p: IntArray, x: Int): Int {
        var y = p[0]
        for (i in 1 until p.size) y = mul(y, x) xor p[i]
        return y
    }

    private fun generator(nsym: Int): IntArray {
        var g = intArrayOf(1)
        for (i in 0 until nsym) g = polyMul(g, intArrayOf(1, pow(2, i)))
        return g
    }

    fun encode(msg: ByteArray, nsym: Int): ByteArray {
        val gen = generator(nsym)
        val out = IntArray(msg.size + nsym)
        for (i in msg.indices) out[i] = msg[i].toInt() and 0xff
        for (i in msg.indices) {
            val coef = out[i]
            if (coef != 0) for (j in 1 until gen.size) out[i + j] = out[i + j] xor mul(gen[j], coef)
        }
        for (i in msg.indices) out[i] = msg[i].toInt() and 0xff
        return ByteArray(out.size) { out[it].toByte() }
    }

    private fun syndromes(msg: IntArray, nsym: Int) = IntArray(nsym + 1) { if (it == 0) 0 else polyEval(msg, pow(2, it - 1)) }

    private fun errataLocator(coefPos: List<Int>): IntArray {
        var loc = intArrayOf(1)
        for (i in coefPos) loc = polyMul(loc, polyAdd(intArrayOf(1), intArrayOf(pow(2, i), 0)))
        return loc
    }

    private fun errorEvaluator(synd: IntArray, errLoc: IntArray, nsym: Int): IntArray {
        val prod = polyMul(synd, errLoc)
        return prod.copyOfRange(maxOf(0, prod.size - (nsym + 1)), prod.size)
    }

    private fun correctErrata(msg: IntArray, synd: IntArray, errPos: List<Int>): IntArray {
        val coefPos = errPos.map { msg.size - 1 - it }
        val errLoc = errataLocator(coefPos)
        val errEval = errorEvaluator(synd.reversedArray(), errLoc, errLoc.size - 1).reversedArray()
        val x = coefPos.map { pow(2, -(255 - it)) }
        val e = IntArray(msg.size)
        for ((i, xi) in x.withIndex()) {
            val xiInv = inv(xi)
            var locPrime = 1
            for (j in x.indices) if (j != i) locPrime = mul(locPrime, 1 xor mul(xiInv, x[j]))
            if (locPrime == 0) throw RsException("could not find error magnitude")
            val y = mul(xi, polyEval(errEval.reversedArray(), xiInv))
            e[errPos[i]] = div(y, locPrime)
        }
        return polyAdd(msg, e)
    }

    private fun errorLocator(synd: IntArray, nsym: Int, eraseCount: Int): IntArray {
        var errLoc = intArrayOf(1)
        var oldLoc = intArrayOf(1)
        val shift = if (synd.size > nsym) synd.size - nsym else 0
        for (i in 0 until nsym - eraseCount) {
            val k = i + shift
            var delta = synd[k]
            for (j in 1 until errLoc.size) delta = delta xor mul(errLoc[errLoc.size - 1 - j], synd[k - j])
            oldLoc = oldLoc + 0
            if (delta != 0) {
                if (oldLoc.size > errLoc.size) {
                    val newLoc = polyScale(oldLoc, delta)
                    oldLoc = polyScale(errLoc, inv(delta))
                    errLoc = newLoc
                }
                errLoc = polyAdd(errLoc, polyScale(oldLoc, delta))
            }
        }
        val first = errLoc.indexOfFirst { it != 0 }
        if (first > 0) errLoc = errLoc.copyOfRange(first, errLoc.size)
        val errs = errLoc.size - 1
        if ((errs - eraseCount) * 2 + eraseCount > nsym) throw RsException("too many errors")
        return errLoc
    }

    private fun findErrors(errLoc: IntArray, n: Int): List<Int> {
        val pos = (0 until n).filter { polyEval(errLoc, pow(2, it)) == 0 }.map { n - 1 - it }
        if (pos.size != errLoc.size - 1) throw RsException("chien search failed")
        return pos
    }

    private fun forneySyndromes(synd: IntArray, pos: List<Int>, n: Int): IntArray {
        val f = synd.copyOfRange(1, synd.size)
        for (p in pos) {
            val x = pow(2, n - 1 - p)
            for (j in 0 until f.size - 1) f[j] = mul(f[j], x) xor f[j + 1]
        }
        return f
    }

    /** Corrected message bytes (parity stripped). Throws [RsException] when beyond repair. */
    fun decode(code: ByteArray, nsym: Int, erasures: List<Int> = emptyList()): ByteArray {
        var msg = IntArray(code.size) { code[it].toInt() and 0xff }
        if (erasures.size > nsym) throw RsException("too many erasures")
        for (e in erasures) msg[e] = 0
        var synd = syndromes(msg, nsym)
        if (synd.any { it != 0 }) {
            val fsynd = forneySyndromes(synd, erasures, msg.size)
            val errLoc = errorLocator(fsynd, nsym, erasures.size)
            val errPos = findErrors(errLoc.reversedArray(), msg.size)
            msg = correctErrata(msg, synd, erasures + errPos)
            synd = syndromes(msg, nsym)
            if (synd.any { it != 0 }) throw RsException("could not correct")
        }
        return ByteArray(msg.size - nsym) { msg[it].toByte() }
    }
}
