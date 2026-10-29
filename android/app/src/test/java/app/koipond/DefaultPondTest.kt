package app.koipond

import app.koipond.core.KoiCodec
import org.junit.Assert.assertEquals
import org.junit.Test
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/** Writes the bundled first-launch pond (assets/default_pond.png): "Hello world", verified to read back. */
class DefaultPondTest {
    @Test
    fun writeDefaultPond() {
        val img = KoiCodec.render("Hello world", 2026L)
        assertEquals("Hello world", KoiCodec.read(img).text)
        val b = BufferedImage(img.w, img.h, BufferedImage.TYPE_INT_RGB)
        b.setRGB(0, 0, img.w, img.h, img.px, 0, img.w)
        val out = File(System.getProperty("user.dir"), "src/main/assets/default_pond.png")
        out.parentFile.mkdirs()
        ImageIO.write(b, "png", out)
        println("default pond: ${out.length() / 1024} KB")
    }
}
