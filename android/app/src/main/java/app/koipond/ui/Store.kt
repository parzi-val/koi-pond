package app.koipond.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** One released pond: its message and when it was released. */
data class PondItem(val id: String, val msg: String, val ts: Long)

/**
 * What the app remembers on this phone: appearance settings, the river of released ponds
 * (newest first, at most 12, as in the design) and which pond is current. Pond images live in
 * filesDir/ponds as full-size PNGs (shared and saved as-is, so they stay decodable) plus thumbnails.
 */
class Store(private val ctx: Context) {
    private val prefs = ctx.getSharedPreferences("koipond", Context.MODE_PRIVATE)
    private val dir = File(ctx.filesDir, "ponds").apply { mkdirs() }

    var theme: String
        get() = prefs.getString("theme", "light")!!
        set(v) = prefs.edit().putString("theme", v).apply()
    var match: Boolean
        get() = prefs.getBoolean("match", false)
        set(v) = prefs.edit().putBoolean("match", v).apply()
    var keep: Boolean
        get() = prefs.getBoolean("keep", true)
        set(v) = prefs.edit().putBoolean("keep", v).apply()

    var history: List<PondItem>
        get() {
            val a = JSONArray(prefs.getString("history", "[]"))
            return (0 until a.length()).map { a.getJSONObject(it).let { o -> PondItem(o.getString("id"), o.getString("msg"), o.getLong("ts")) } }
        }
        set(v) {
            val a = JSONArray()
            v.forEach { a.put(JSONObject().put("id", it.id).put("msg", it.msg).put("ts", it.ts)) }
            prefs.edit().putString("history", a.toString()).apply()
            val keepIds = v.map { it.id }.toSet() + current.id
            dir.listFiles()?.forEach { f -> if (f.name.substringBefore('.').removeSuffix("_t") !in keepIds) f.delete() }
        }

    var current: PondItem
        get() = prefs.getString("current", null)?.let { JSONObject(it) }
            ?.let { PondItem(it.getString("id"), it.getString("msg"), it.getLong("ts")) } ?: defaultPond()
        set(v) = prefs.edit().putString("current", JSONObject().put("id", v.id).put("msg", v.msg).put("ts", v.ts).toString()).apply()

    fun full(id: String) = File(dir, "$id.png")
    private fun thumb(id: String) = File(dir, "${id}_t.jpg")

    /** The bundled first-launch pond ("Hello world", rendered and verified by the same core). */
    private fun defaultPond(): PondItem {
        val item = PondItem("default", "Hello world", System.currentTimeMillis())
        if (!full(item.id).exists()) ctx.assets.open("default_pond.png").use { i -> full(item.id).outputStream().use { i.copyTo(it) } }
        return item
    }

    fun save(id: String, bmp: Bitmap) {
        full(id).outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        thumb(id).outputStream().use { Bitmap.createScaledBitmap(bmp, 360, 360, true).compress(Bitmap.CompressFormat.JPEG, 90, it) }
    }

    fun loadFull(id: String): Bitmap? = full(id).takeIf { it.exists() }?.let { BitmapFactory.decodeFile(it.path) }

    fun loadThumb(id: String): Bitmap? {
        val t = thumb(id)
        if (!t.exists()) loadFull(id)?.let { b -> t.outputStream().use { Bitmap.createScaledBitmap(b, 360, 360, true).compress(Bitmap.CompressFormat.JPEG, 90, it) } }
        return t.takeIf { it.exists() }?.let { BitmapFactory.decodeFile(it.path) }
    }
}
