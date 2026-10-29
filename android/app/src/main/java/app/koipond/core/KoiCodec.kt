package app.koipond.core

/** Text in, pond out; pond in, text out. Payload = UTF-8 text zero-padded to PAYLOAD_BYTES, outer RS on top. */
object KoiCodec {
    val MAX_BYTES = KoiSpec.PAYLOAD_BYTES

    fun bytesOf(text: String): ByteArray = text.toByteArray(Charsets.UTF_8)

    fun codeword(text: String): ByteArray {
        val b = bytesOf(text)
        require(b.size <= MAX_BYTES) { "message is ${b.size} bytes; the pond holds $MAX_BYTES" }
        return Rs.encode(b.copyOf(MAX_BYTES), KoiSpec.PARITY_BYTES)
    }

    /** Render, then read it back; re-roll the style (never the message) if the clean pond misreads. */
    fun render(text: String, seed: Long, onAttempt: (Int) -> Unit = {}): Rgb {
        val code = codeword(text)
        var img: Rgb? = null
        for (attempt in 0 until 4) {
            onAttempt(attempt)
            img = Prof.sec("render") { KoiRender.render(code, seed * 7919 + attempt) }
            val r = Prof.sec("verify") { KoiDecode.decode(img) }
            if (r != null && r.erased.isEmpty() && r.codeword.contentEquals(code)) return img
        }
        return img!!
    }

    /** One pond for this message and style (no read-back). */
    fun renderOnce(text: String, seed: Long, attempt: Int = 0): Rgb = KoiRender.render(codeword(text), seed * 7919 + attempt)

    /** Does this pond read back to exactly this message with every koi intact? */
    fun verifies(img: Rgb, text: String): Boolean {
        val r = KoiDecode.decode(img) ?: return false
        return r.erased.isEmpty() && r.codeword.contentEquals(codeword(text))
    }

    class Scan(val text: String?, val result: KoiDecode.Result?)

    private fun attempt(img: Rgb, fullRes: Boolean, onKoi: (List<KoiDecode.KoiRead>, Int) -> Unit): Scan {
        val r = KoiDecode.decode(img, fullRes, onKoi) ?: return Scan(null, null)
        return try {
            val msg = Rs.decode(r.codeword, KoiSpec.PARITY_BYTES, r.erased)
            val end = msg.indexOfLast { it.toInt() != 0 } + 1
            Scan(String(msg, 0, end, Charsets.UTF_8), r)
        } catch (e: RsException) {
            Scan(null, r)
        }
    }

    /** Fast pass first; if the message doesn't come back, look again with full-resolution silhouettes. */
    fun read(img: Rgb, onKoi: (List<KoiDecode.KoiRead>, Int) -> Unit = { _, _ -> }): Scan {
        val fast = attempt(img, false, onKoi)
        if (fast.text != null || minOf(img.w, img.h) < 1600) return fast
        val slow = attempt(img, true, onKoi)
        return if (slow.text != null || (slow.result?.read ?: 0) > (fast.result?.read ?: 0)) slow else fast
    }
}
