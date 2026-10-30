package app.koipond.ui

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.ImageCapture
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import app.koipond.BuildConfig
import app.koipond.core.Anchors
import app.koipond.core.KoiCodec
import app.koipond.core.Prof
import app.koipond.core.StripReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.math.max
import kotlin.math.roundToInt

private const val KEEP_FRAMES = 12  // ~2-3 MB each: the zip stays small enough to send from the phone

/** Koi Lab: scan continuously, log what the decoder saw for every frame, keep the last frames; copy the log, zip the frames. */
@Composable
fun LabScreen() {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val ask = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    val capture = remember { ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build() }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var running by remember { mutableStateOf(true) }
    var status by remember { mutableStateOf("starting...") }
    val dir = remember { File(ctx.cacheDir, "lab").apply { mkdirs() } }
    // the log lives in a file too: switching apps (to paste it somewhere) can recreate this screen
    val logFile = remember { File(ctx.filesDir, "lab_log.txt") }
    val log = remember { mutableStateListOf<String>().apply { if (logFile.exists()) addAll(logFile.readLines().filter { it.isNotBlank() }) } }
    val stamp = remember { SimpleDateFormat("HH:mm:ss", Locale.US) }
    fun header() = "Koi Lab ${BuildConfig.VERSION_NAME} | ${Build.MANUFACTURER} ${Build.MODEL} Android ${Build.VERSION.RELEASE} | " +
        "${Runtime.getRuntime().availableProcessors()} cores | reader ${if (StripReader.ready) "learned (loaded in ${StripReader.loadMs} ms)" else "classical"}"

    LaunchedEffect(Unit) { if (!granted) ask.launch(Manifest.permission.CAMERA) }
    LaunchedEffect(granted, camera, running) {
        if (!granted || camera == null || !running) return@LaunchedEffect
        delay(700)
        var n = log.size
        while (isActive) {
            val bmp = try { capture.grab(ctx) } catch (e: Exception) { delay(800); continue }
            n++
            val t0 = System.nanoTime()
            val rgb = withContext(Dispatchers.Default) { bmp.toRgb(3000) }
            Prof.ms.clear()
            val scan = withContext(Dispatchers.Default) { KoiCodec.read(rgb) }
            val ms = (System.nanoTime() - t0) / 1_000_000
            val r = scan.result
            val line = "#$n ${stamp.format(Date())} ${rgb.w}x${rgb.h} bullseyes ${Anchors.lastFound} " +
                "${if (r?.anchored == true) "unwarped" else "not unwarped"} | koi found ${r?.kois?.size ?: 0} read ${r?.read ?: 0}/6 | " +
                "${if (scan.text != null) "MESSAGE \"${scan.text}\"" else "no message"} | ${ms} ms | ${Prof.report()}"
            log += line
            withContext(Dispatchers.IO) { logFile.appendText(line + "\n") }
            status = "#$n  bullseyes ${Anchors.lastFound}  koi ${r?.read ?: 0}/6  ${if (scan.text != null) "✓ \"${scan.text}\"" else "no message"}  ${ms} ms"
            withContext(Dispatchers.IO) {  // the decoder's input, as JPEG; keep the newest KEEP_FRAMES
                val s = 3000f / max(bmp.width, bmp.height)
                val b = if (s < 1f) Bitmap.createScaledBitmap(bmp, (bmp.width * s).roundToInt(), (bmp.height * s).roundToInt(), true) else bmp
                File(dir, "frame_%04d.jpg".format(n)).outputStream().use { b.compress(Bitmap.CompressFormat.JPEG, 90, it) }
                dir.listFiles()!!.sortedBy { it.name }.dropLast(KEEP_FRAMES).forEach { it.delete() }
            }
        }
    }

    Column(Modifier.fillMaxSize().background(Color(0xFF101418)).statusBarsPadding().navigationBarsPadding()) {
        Box(Modifier.fillMaxWidth().weight(1f)) {
            if (granted) CameraPreview(capture) { camera = it }
            else Text("Koi Lab needs the camera.", color = Color.White, modifier = Modifier.align(Alignment.Center))
            Text(status, color = Color.White, fontSize = 15.sp, modifier = Modifier.align(Alignment.BottomStart)
                .fillMaxWidth().background(Color(0xAA000000)).padding(10.dp))
        }
        Row(Modifier.fillMaxWidth().padding(8.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Button({ running = !running }, Modifier.weight(1f)) { Text(if (running) "Pause" else "Resume", fontSize = 13.sp) }
            Button({ copyLog(ctx, (listOf(header()) + log).joinToString("\n")) }, Modifier.weight(1f)) { Text("Copy log", fontSize = 13.sp) }
            Button({
                scope.launch {
                    val name = withContext(Dispatchers.IO) { runCatching { saveZip(ctx, dir, (listOf(header()) + log).joinToString("\n")) } }
                    Toast.makeText(ctx, name.fold({ "Saved Download/$it" }, { "Zip failed: ${it.message}" }), Toast.LENGTH_LONG).show()
                }
            }, Modifier.weight(1f)) { Text("Save zip", fontSize = 13.sp) }
            Button({ log.clear(); logFile.delete(); dir.listFiles()?.forEach { it.delete() }; status = "cleared" }, Modifier.weight(1f)) { Text("Clear", fontSize = 13.sp) }
        }
        Column(Modifier.fillMaxWidth().weight(.8f).verticalScroll(rememberScrollState(), reverseScrolling = true).padding(horizontal = 8.dp)) {
            Text(header(), color = Color(0xFF9FB3C8), fontSize = 11.sp, fontFamily = FontFamily.Monospace)
            for (l in log.takeLast(60)) Text(l, color = Color(0xFFE0E6EB), fontSize = 11.sp, fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(top = 4.dp))
        }
    }
}

private fun copyLog(ctx: android.content.Context, text: String) {
    copyText(ctx, text)
    Toast.makeText(ctx, "Log copied (${text.lines().size} lines)", Toast.LENGTH_SHORT).show()
}

/** The kept frames + the log as Download/koi-lab-<time>.zip; returns the file name. */
private fun saveZip(ctx: android.content.Context, dir: File, log: String): String {
    val name = "koi-lab-${SimpleDateFormat("MMdd-HHmmss", Locale.US).format(Date())}.zip"
    val values = ContentValues().apply {
        put(MediaStore.Downloads.DISPLAY_NAME, name)
        put(MediaStore.Downloads.MIME_TYPE, "application/zip")
        put(MediaStore.Downloads.IS_PENDING, 1)
    }
    val r = ctx.contentResolver
    val uri = r.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: error("could not create the file")
    r.openOutputStream(uri)!!.use { o ->
        ZipOutputStream(o).use { z ->
            z.setLevel(0)  // JPEGs don't shrink
            z.putNextEntry(ZipEntry("log.txt")); z.write(log.toByteArray()); z.closeEntry()
            for (f in dir.listFiles()!!.sortedBy { it.name }) { z.putNextEntry(ZipEntry(f.name)); f.inputStream().use { it.copyTo(z) }; z.closeEntry() }
        }
    }
    values.clear()
    values.put(MediaStore.Downloads.IS_PENDING, 0)
    r.update(uri, values, null, null)
    return name
}
