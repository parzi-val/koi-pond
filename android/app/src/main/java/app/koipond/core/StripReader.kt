package app.koipond.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/** The learned koi reader (research/nn/koinn.pt, exported by export_koi.py): each koi unbent into a strip, 176 bits out. */
object StripReader {
    const val SH = 128
    const val SW = 640
    private const val V_SPAN = 1.15

    @Volatile private var session: OrtSession? = null  // loaded on a background thread
    val ready get() = session != null

    /** Load once with koinn.onnx's bytes; until then KoiDecode reads the hatches classically. */
    fun load(model: ByteArray) {
        if (session == null) session = OrtEnvironment.getEnvironment().createSession(model, OrtSession.SessionOptions())
    }

    /** koi_data.strip: u (nose -> tail) along the width, v in +-V_SPAN body half-widths down the height. */
    fun strip(ink: FImg, fr: KoiDecode.Frame, out: FloatArray, at: Int) {
        for (r in 0 until SH) {
            val v = -V_SPAN + 2 * V_SPAN * r / (SH - 1)
            for (c in 0 until SW) out[at + r * SW + c] = fr.sample(ink, c.toDouble() / (SW - 1), v)
        }
    }

    /** One strip per frame, read as photographed and v-flipped (a mirrored photo) -> 2 bit sets per frame. */
    fun read(ink: FImg, frames: List<KoiDecode.Frame>): List<Pair<IntArray, IntArray>> {
        val n = frames.size
        val px = SH * SW
        val buf = FloatArray(2 * n * px)
        for ((i, fr) in frames.withIndex()) {
            strip(ink, fr, buf, 2 * i * px)
            for (r in 0 until SH) System.arraycopy(buf, 2 * i * px + r * SW, buf, (2 * i + 1) * px + (SH - 1 - r) * SW, SW)
        }
        val env = OrtEnvironment.getEnvironment()
        val logits = OnnxTensor.createTensor(env, FloatBuffer.wrap(buf), longArrayOf(2L * n, 1, SH.toLong(), SW.toLong())).use { x ->
            session!!.run(mapOf("ink" to x)).use { (it[0].value as Array<FloatArray>) }
        }
        return List(n) { i -> Pair(IntArray(logits[2 * i].size) { if (logits[2 * i][it] > 0) 1 else 0 },
                                  IntArray(logits[2 * i + 1].size) { if (logits[2 * i + 1][it] > 0) 1 else 0 }) }
    }
}
