package app.koipond

import app.koipond.core.KoiCodec
import app.koipond.core.KoiDecode
import app.koipond.core.KoiSpec
import app.koipond.core.Rgb
import app.koipond.core.Rs
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/** The Kotlin port must speak exactly the format of koi.py, both directions (fixtures written by Python). */
class KoiTest {
    private val fx = File(System.getProperty("user.dir"), "../fixtures")
    private fun hex(s: String) = ByteArray(s.length / 2) { s.substring(2 * it, 2 * it + 2).toInt(16).toByte() }
    private fun load(name: String): Rgb {
        val b = ImageIO.read(File(fx, name))
        val px = IntArray(b.width * b.height)
        b.getRGB(0, 0, b.width, b.height, px, 0, b.width)
        return Rgb(b.width, b.height, IntArray(px.size) { px[it] and 0xffffff })
    }
    private fun save(img: Rgb, name: String) {
        val b = BufferedImage(img.w, img.h, BufferedImage.TYPE_INT_RGB)
        b.setRGB(0, 0, img.w, img.h, img.px, 0, img.w)
        ImageIO.write(b, "png", File(fx, name))
    }

    @Test
    fun cellGridMatchesPython() {
        val (cols, consts) = File(fx, "cells.txt").readLines()
        val counts = IntArray(KoiSpec.COLS) { c -> KoiSpec.STARTS[c + 1] - KoiSpec.STARTS[c] }
        assertEquals(cols, counts.joinToString(" "))
        assertEquals(consts, "${KoiSpec.FISH_BITS} ${KoiSpec.CHUNK} ${KoiSpec.PAYLOAD_BYTES} ${KoiSpec.PARITY_BYTES}")
    }

    @Test
    fun readsPythonPonds() {
        val code = hex(File(fx, "py_pond.txt").readText().trim())
        var t = System.currentTimeMillis()
        val r = KoiDecode.decode(load("py_pond.png"))!!
        println("python pond: ${r.read}/${r.kois.size} koi read, erased ${r.erased.size}, ${System.currentTimeMillis() - t} ms | ${app.koipond.core.Prof.report()}")
        assertTrue(r.erased.isEmpty())
        assertArrayEquals(code, r.codeword)
        t = System.currentTimeMillis()
        val r2 = KoiDecode.decode(load("py_pond_photo1.jpg"))!!
        println("python pond, photo-1 distorted: ${r2.read}/${r2.kois.size} koi read, ${System.currentTimeMillis() - t} ms")
        val payload = code.copyOf(KoiSpec.PAYLOAD_BYTES)
        assertArrayEquals(payload, Rs.decode(r2.codeword, KoiSpec.PARITY_BYTES, r2.erased))
    }

    @Test
    fun kotlinPondRoundTrips() {
        val text = "Koi carry messages in their scales 🐟"
        var t = System.currentTimeMillis()
        val img = KoiCodec.render(text, 42L)
        println("render (incl. verify): ${System.currentTimeMillis() - t} ms | ${app.koipond.core.Prof.report()}")
        save(img, "kt_pond.png")
        File(fx, "kt_pond.txt").writeText(KoiCodec.codeword(text).joinToString("") { "%02x".format(it) })
        t = System.currentTimeMillis()
        val s = KoiCodec.read(img)
        println("read: '${s.text}' (${s.result?.read}/${s.result?.kois?.size} koi) ${System.currentTimeMillis() - t} ms | ${app.koipond.core.Prof.report()}")
        assertEquals(text, s.text)
    }
}
