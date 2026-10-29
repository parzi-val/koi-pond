package app.koipond.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.provider.MediaStore
import androidx.core.content.FileProvider
import app.koipond.core.Rgb
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

fun Rgb.toBitmap(): Bitmap {
    val p = IntArray(px.size) { px[it] or 0xFF000000.toInt() }
    return Bitmap.createBitmap(p, w, h, Bitmap.Config.ARGB_8888)
}

/** Pixels for the decoder, shrunk so the long side is at most maxSide (the decoder is scale-tolerant). */
fun Bitmap.toRgb(maxSide: Int = 3000): Rgb {
    val s = maxSide.toFloat() / max(width, height)
    val b = if (s < 1f) Bitmap.createScaledBitmap(this, (width * s).roundToInt(), (height * s).roundToInt(), true) else this
    val px = IntArray(b.width * b.height)
    b.getPixels(px, 0, b.width, 0, 0, b.width, b.height)
    for (i in px.indices) px[i] = px[i] and 0xffffff
    return Rgb(b.width, b.height, px)
}

fun Bitmap.rotated(deg: Int): Bitmap =
    if (deg % 360 == 0) this else Bitmap.createBitmap(this, 0, 0, width, height, Matrix().apply { postRotate(deg.toFloat()) }, true)

private fun uriOf(ctx: Context, file: File) = FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", file)

/** The system share sheet, with the pond PNG exactly as stored (so it stays readable). */
fun sharePond(ctx: Context, file: File) {
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "image/png"
        putExtra(Intent.EXTRA_STREAM, uriOf(ctx, file))
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(Intent.createChooser(send, "Share this pond").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
}

/** Into Photos (Pictures/Koi Pond). */
fun savePond(ctx: Context, file: File, name: String) {
    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, "$name.png")
        put(MediaStore.Images.Media.MIME_TYPE, "image/png")
        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Koi Pond")
        put(MediaStore.Images.Media.IS_PENDING, 1)
    }
    val r = ctx.contentResolver
    val uri = r.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("could not create image")
    r.openOutputStream(uri)?.use { o -> file.inputStream().use { it.copyTo(o) } }
    values.clear()
    values.put(MediaStore.Images.Media.IS_PENDING, 0)
    r.update(uri, values, null, null)
}

fun copyPondImage(ctx: Context, file: File) {
    val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    cm.setPrimaryClip(ClipData.newUri(ctx.contentResolver, "Koi pond", uriOf(ctx, file)))
}

fun copyText(ctx: Context, text: String) {
    val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    cm.setPrimaryClip(ClipData.newPlainText("koi message", text))
}
