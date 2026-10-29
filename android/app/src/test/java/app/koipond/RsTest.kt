package app.koipond

import app.koipond.core.Rs
import org.junit.Assert.assertArrayEquals
import org.junit.Test
import java.io.File

/** Kotlin RS must match Python reedsolo byte for byte (fixtures/rs.txt, written by reedsolo). */
class RsTest {
    private fun hex(s: String) = ByteArray(s.length / 2) { s.substring(2 * it, 2 * it + 2).toInt(16).toByte() }

    @Test
    fun matchesReedsolo() {
        val lines = File(System.getProperty("user.dir"), "../fixtures/rs.txt").readLines()
        for (l in lines) {
            val (nsym, msg, code, bad, er) = l.split(" ")
            val n = nsym.toInt()
            assertArrayEquals("encode $l", hex(code), Rs.encode(hex(msg), n))
            val erasures = if (er == "-") emptyList() else er.split(",").map { it.toInt() }
            assertArrayEquals("decode $l", hex(msg), Rs.decode(hex(bad), n, erasures))
        }
        println("rs vectors ok: ${lines.size}")
    }
}
