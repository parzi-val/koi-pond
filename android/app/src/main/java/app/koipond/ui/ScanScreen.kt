package app.koipond.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Matrix
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import app.koipond.core.KoiCodec
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.max

/** One full-resolution frame, upright. */
private suspend fun ImageCapture.grab(ctx: Context): Bitmap = suspendCancellableCoroutine { cont ->
    takePicture(ContextCompat.getMainExecutor(ctx), object : ImageCapture.OnImageCapturedCallback() {
        override fun onCaptureSuccess(image: ImageProxy) {
            val b = image.toBitmap().rotated(image.imageInfo.rotationDegrees)
            image.close()
            cont.resume(b)
        }
        override fun onError(exception: ImageCaptureException) = cont.resumeWithException(exception)
    })
}

private val sixWords = listOf("no", "one", "two", "three", "four", "five", "six")

/** The design's four rounded corner brackets (340-unit frame), as paths. */
private val BRACKETS = listOf("M2 38V16Q2 2 16 2H38", "M302 2H324Q338 2 338 16V38", "M338 302V324Q338 338 324 338H302", "M38 338H16Q2 338 2 324V302")

@Composable
fun ScanScreen(app: AppState, scope: CoroutineScope) {
    val ctx = LocalContext.current
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val ask = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    val capture = remember { ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build() }
    var camera by remember { mutableStateOf<Camera?>(null) }
    LaunchedEffect(Unit) { if (!granted) ask.launch(Manifest.permission.CAMERA) }
    LaunchedEffect(app.torch, camera) { camera?.cameraControl?.enableTorch(app.torch) }

    // read frame after frame until a pond gives up its message (no shutter, as designed)
    LaunchedEffect(granted, app.scanFound, camera) {
        if (!granted || app.scanFound || camera == null) return@LaunchedEffect
        delay(700)
        while (isActive) {
            val bmp = try { capture.grab(ctx) } catch (e: Exception) { Log.w("koipond", "capture", e); delay(800); continue }
            val rgb = withContext(Dispatchers.Default) { bmp.toRgb(3000) }
            val scan = withContext(Dispatchers.Default) {
                KoiCodec.read(rgb) { reads, _ ->
                    val read = reads.filter { it.index != null }
                    scope.launch(Dispatchers.Main) {
                        app.scanRead = read.size
                        app.scanSize = rgb.w to rgb.h
                        app.scanDots = read.mapNotNull { it.spine?.let { s -> Offset(s.x[0].toFloat(), s.y[0].toFloat()) } }
                    }
                }
            }
            if (scan.text != null) {
                app.scanText = scan.text
                app.scanAll = scan.result?.read ?: 6
                app.scanFound = true
                break
            }
        }
    }

    BoxWithConstraints(Modifier.fillMaxSize().background(Cam.bg)) {
        val wPx = with(LocalDensity.current) { maxWidth.toPx() }
        val hPx = with(LocalDensity.current) { maxHeight.toPx() }
        if (granted) CameraPreview(capture) { camera = it } else NoCamera { ask.launch(Manifest.permission.CAMERA) }

        val bracket by animateColorAsState(if (app.scanFound) Cam.found else Cam.bracket, label = "bracket")
        Canvas(Modifier.offset(x = (maxWidth - 340.dp) / 2, y = 170.dp).size(340.dp)) {
            val m = Matrix().apply { scale(size.width / 340f, size.width / 340f) }
            val st = Stroke(3.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round)
            for (d in BRACKETS) drawPath(Path().apply { addPath(G.path(d)); transform(m) }, bracket, style = st)
        }
        // a dot on each koi as it is read (capture pixels -> preview, which fills the screen centre-cropped)
        if (app.scanSize.first > 0) Canvas(Modifier.fillMaxSize()) {
            val (iw, ih) = app.scanSize
            val k = max(wPx / iw, hPx / ih)
            val ox = (wPx - iw * k) / 2
            val oy = (hPx - ih * k) / 2
            for (p in app.scanDots) {
                val c = Offset(ox + p.x * k, oy + p.y * k)
                drawCircle(Color.White.copy(alpha = .6f), 11.dp.toPx(), c)
                drawCircle(Cam.accent, 8.dp.toPx(), c)
            }
        }

        Row(Modifier.fillMaxWidth().statusBarsPadding().padding(start = 20.dp, end = 20.dp, top = 8.dp)) {
            Glass(G.CLOSE, 1.8f, pressed = false) { app.go(Tab.CREATE) }
            Spacer(Modifier.weight(1f))
            Glass(G.FLASH, 1.7f, pressed = app.torch) { app.torch = !app.torch }
        }
        if (!app.scanFound) {
            Box(Modifier.fillMaxWidth().offset(y = 124.dp), contentAlignment = Alignment.Center) {
                Text("Fit the whole pond in view", style = Type.hint.copy(color = Cam.ink),
                    modifier = Modifier.clip(RoundedCornerShape(18.dp)).background(Cam.glass).padding(horizontal = 16.dp, vertical = 8.dp))
            }
            Column(Modifier.fillMaxWidth().offset(y = 610.dp), horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("reading the koi…", style = Type.reading.copy(color = Cam.ink))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) { for (i in 0 until 6) FishGlyph(i < app.scanRead, Cam.sub, Cam.accent) }
                Text("${app.scanRead} of 6 koi read", style = Type.small.copy(color = Cam.sub))
            }
        } else {
            FoundSheet(app, ctx, scope, Modifier.align(Alignment.BottomCenter))
        }
    }
}

@Composable
private fun CameraPreview(capture: ImageCapture, onCamera: (Camera) -> Unit) {
    val ctx = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val view = remember { PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }
    DisposableEffect(owner) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(view.surfaceProvider) }
            provider.unbindAll()
            onCamera(provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture))
        }, ContextCompat.getMainExecutor(ctx))
        onDispose { runCatching { future.get().unbindAll() } }
    }
    AndroidView({ view }, Modifier.fillMaxSize())
}

@Composable
private fun NoCamera(onAllow: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(40.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Text("The scanner needs your camera to see a pond.", style = Type.reading.copy(color = Cam.ink), textAlign = TextAlign.Center)
        Spacer(Modifier.height(20.dp))
        Box(Modifier.height(54.dp).clip(RoundedCornerShape(27.dp)).border(1.5.dp, Cam.ink, RoundedCornerShape(27.dp)).clickable { onAllow() }
            .padding(horizontal = 28.dp), contentAlignment = Alignment.Center) {
            Text("Allow camera", style = Type.pill.copy(color = Cam.ink))
        }
    }
}

/** Frosted round button over the camera (close, flashlight). */
@Composable
private fun Glass(d: String, stroke: Float, pressed: Boolean, onClick: () -> Unit) {
    Box(Modifier.size(44.dp).clip(CircleShape).background(if (pressed) Cam.ink else Cam.glass).clickable { onClick() },
        contentAlignment = Alignment.Center) { Icon24(d, if (pressed) Cam.bg else Cam.ink, 22.dp, stroke) }
}

@Composable
private fun FoundSheet(app: AppState, ctx: Context, scope: CoroutineScope, modifier: Modifier) {
    val c = LocalPond.current
    Column(
        modifier.fillMaxWidth().clip(RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp)).background(c.sheet).navigationBarsPadding()
            .padding(start = 28.dp, end = 28.dp, top = 14.dp, bottom = 28.dp),
        verticalArrangement = Arrangement.spacedBy(22.dp)
    ) {
        Box(Modifier.align(Alignment.CenterHorizontally).size(40.dp, 4.dp).clip(RoundedCornerShape(2.dp)).background(c.handle))
        Column {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Icon24(G.CHECK, c.ok, 14.dp, 2.4f)
                Text("FOUND IN THE POND", style = Type.eyebrow.copy(color = c.sub))
            }
            Text(app.scanText, style = Type.found.copy(color = c.ink), modifier = Modifier.padding(top = 8.dp))
            val n = app.scanAll
            Text("${KoiCodec.bytesOf(app.scanText).size} bytes · " + (if (n >= 6) "all six koi read back" else "${sixWords[n]} of six koi read back"),
                style = Type.small.copy(color = c.sub), modifier = Modifier.padding(top = 8.dp))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Pill("Copy", Modifier.weight(1f), solid = true, icon = G.COPY) { copyText(ctx, app.scanText); app.say(scope, "Message copied") }
            Pill("Scan again", Modifier.weight(1f), solid = false) { app.resetScan() }
        }
    }
}

/** The sheets' pill buttons: solid ink, or outlined. */
@Composable
fun Pill(text: String, modifier: Modifier = Modifier, solid: Boolean, icon: String? = null, onClick: () -> Unit) {
    val c = LocalPond.current
    val fg = if (solid) c.sheet else c.ink
    Row(
        modifier.height(54.dp).clip(RoundedCornerShape(27.dp)).background(if (solid) c.ink else Color.Transparent)
            .border(1.5.dp, c.ink, RoundedCornerShape(27.dp)).clickable { onClick() },
        horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically
    ) {
        if (icon != null) { Icon24(icon, fg, 20.dp, 1.7f); Spacer(Modifier.width(10.dp)) }
        Text(text, style = Type.pill.copy(color = fg))
    }
}
