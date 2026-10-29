package app.koipond

import app.koipond.core.KoiDecode
import app.koipond.core.Prof
import app.koipond.core.Rgb
import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Decoder benchmark for the phone, no UI or camera: run from adb with the APK as classpath,
 *   adb shell CLASSPATH=/data/local/tmp/koi.apk app_process / app.koipond.Bench /data/local/tmp/bench [runs]
 * Every NAME.rgb in the folder (int32 LE w, h, then w*h RGB bytes) is decoded `runs` times after one
 * warm-up, and checked against NAME.hex (the expected outer codeword).
 */
object Bench {
    @JvmStatic
    fun main(args: Array<String>) {
        val dir = File(args[0])
        val runs = args.getOrNull(1)?.toInt() ?: 3
        println("cores ${Runtime.getRuntime().availableProcessors()}  max heap ${Runtime.getRuntime().maxMemory() / 1_000_000} MB")
        for (f in dir.listFiles { p -> p.name.endsWith(".rgb") }!!.sortedBy { it.name }) {
            val img = load(f)
            val want = File(dir, f.name.removeSuffix(".rgb") + ".hex").readText().trim()
            KoiDecode.decode(img); Prof.ms.clear()  // warm-up: let the JIT see the hot loops once
            val total = ArrayList<Long>()
            var stages = ""
            var ok = ""
            repeat(runs) {
                val t = System.nanoTime()
                val r = KoiDecode.decode(img)
                total += (System.nanoTime() - t) / 1_000_000
                stages = Prof.report()
                ok = if (r == null) "no koi" else "${r.read}/6 koi, erased ${r.erased.size}, codeword ${if (hex(r.codeword) == want) "match" else "MISMATCH"}"
            }
            println("${f.name} ${img.w}x${img.h}: ${total.joinToString(" ")} ms  ($ok)")
            println("    last run: $stages")
        }
    }

    private fun load(f: File): Rgb {
        DataInputStream(f.inputStream().buffered()).use { s ->
            val head = ByteArray(8).also { s.readFully(it) }
            val bb = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN)
            val w = bb.int
            val h = bb.int
            val raw = ByteArray(w * h * 3).also { s.readFully(it) }
            return Rgb(w, h, IntArray(w * h) { i ->
                (raw[3 * i].toInt() and 255 shl 16) or (raw[3 * i + 1].toInt() and 255 shl 8) or (raw[3 * i + 2].toInt() and 255)
            })
        }
    }

    private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }
}
