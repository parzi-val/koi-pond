package app.koipond.ui

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.view.WindowCompat
import app.koipond.core.KoiCodec
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.random.Random

enum class Tab { CREATE, SCAN, HISTORY }
enum class Phase { IDLE, DRAWING, CHECKING, FAILED }

/** Everything the screens share. Lives above the tabs so switching never loses work. */
class AppState(ctx: Context) {
    val store = Store(ctx)
    var tab by mutableStateOf(Tab.CREATE)
    var theme by mutableStateOf(store.theme)
    var match by mutableStateOf(store.match)
    var keep by mutableStateOf(store.keep)
    var history by mutableStateOf(store.history)
    var current by mutableStateOf(store.current)
    var pond by mutableStateOf<Bitmap?>(null)
    var msg by mutableStateOf(current.msg)
    var phase by mutableStateOf(Phase.IDLE)
    var drops by mutableIntStateOf(0)        // bumps when a new pond lands: drives the drop + ripple
    var settingsOpen by mutableStateOf(false)
    var shareOpen by mutableStateOf(false)
    var toast by mutableStateOf<String?>(null)
    private var toastId = 0

    // scan: live progress of the current read, and what was found
    var scanFound by mutableStateOf(false)
    var scanText by mutableStateOf("")
    var scanRead by mutableIntStateOf(0)
    var scanAll by mutableIntStateOf(0)
    var scanDots by mutableStateOf<List<androidx.compose.ui.geometry.Offset>>(emptyList())   // read koi heads, in capture pixels
    var scanSize by mutableStateOf(0 to 0)
    var torch by mutableStateOf(false)

    fun resetScan() { scanFound = false; scanText = ""; scanRead = 0; scanAll = 0; scanDots = emptyList(); scanSize = 0 to 0 }

    fun go(t: Tab) {
        if (t == Tab.SCAN && tab != Tab.SCAN) resetScan()
        if (t != Tab.SCAN) torch = false
        tab = t
    }

    val bytes get() = KoiCodec.bytesOf(msg).size
    val dirty get() = msg != current.msg

    fun say(scope: CoroutineScope, text: String) {
        val id = ++toastId
        toast = text
        scope.launch { delay(2600); if (toastId == id) toast = null }
    }

    fun pickTheme(t: String) { theme = t; match = false; store.theme = t; store.match = false }
    fun setMatch(on: Boolean, systemDark: Boolean) {
        match = on; store.match = on
        if (!on) { theme = if (systemDark) "dark" else "light"; store.theme = theme }
    }
    fun pickKeep(on: Boolean) { keep = on; store.keep = on }
    fun clearHistory() { history = emptyList(); store.history = emptyList() }

    fun open(item: PondItem, scope: CoroutineScope) {
        current = item; store.current = item; msg = item.msg; phase = Phase.IDLE; go(Tab.CREATE)
        scope.launch { pond = withContext(Dispatchers.IO) { store.loadFull(item.id) } }
    }

    /** Draw the pond, show it, read it back; redraw in a new style if any koi misreads. */
    fun release(scope: CoroutineScope) {
        val text = msg
        if (text.isBlank() || KoiCodec.bytesOf(text).size > KoiCodec.MAX_BYTES || phase == Phase.DRAWING || phase == Phase.CHECKING) return
        phase = Phase.DRAWING
        scope.launch {
            val seed = Random.nextLong()
            var ok = false
            var bmp: Bitmap? = null
            val item = PondItem("p${System.currentTimeMillis()}", text, System.currentTimeMillis())
            for (attempt in 0 until 3) {
                phase = Phase.DRAWING
                val img = withContext(Dispatchers.Default) { KoiCodec.renderOnce(text, seed, attempt) }
                bmp = withContext(Dispatchers.Default) { img.toBitmap() }
                pond = bmp
                current = item
                drops++
                phase = Phase.CHECKING
                ok = withContext(Dispatchers.Default) { KoiCodec.verifies(img, text) }
                if (ok) break
            }
            withContext(Dispatchers.IO) { store.save(item.id, bmp!!) }
            store.current = item
            if (keep) { history = (listOf(item) + history).take(12); store.history = history }
            phase = if (ok) Phase.IDLE else Phase.FAILED
        }
    }
}

@Composable
fun KoiApp() {
    val ctx = LocalContext.current
    val app = remember { AppState(ctx.applicationContext) }
    val scope = rememberCoroutineScope()
    val systemDark = isSystemInDarkTheme()
    val dark = if (app.match) systemDark else app.theme == "dark"
    val colors = if (dark) NightPond else StillWater
    LaunchedEffect(app.current.id) { if (app.pond == null) app.pond = withContext(Dispatchers.IO) { app.store.loadFull(app.current.id) } }

    val view = LocalView.current
    val lightBars = !dark && app.tab != Tab.SCAN
    SideEffect {
        val w = (view.context as Activity).window
        WindowCompat.getInsetsController(w, view).apply { isAppearanceLightStatusBars = lightBars; isAppearanceLightNavigationBars = lightBars }
    }

    BackHandler(app.shareOpen || app.settingsOpen || app.tab != Tab.CREATE) {
        when {
            app.shareOpen -> app.shareOpen = false
            app.settingsOpen -> app.settingsOpen = false
            else -> app.go(Tab.CREATE)
        }
    }

    CompositionLocalProvider(LocalPond provides colors) {
        Box(Modifier.fillMaxSize().background(if (app.tab == Tab.SCAN) Cam.bg else colors.bg)) {
            when (app.tab) {
                Tab.CREATE -> CreateScreen(app, scope)
                Tab.SCAN -> ScanScreen(app, scope)
                Tab.HISTORY -> HistoryScreen(app, scope)
            }
            if (!(app.tab == Tab.SCAN && app.scanFound)) Nav(app, Modifier.align(Alignment.BottomCenter))
            if (app.settingsOpen) SettingsScreen(app, scope, systemDark)
            if (app.shareOpen) ShareSheet(app, scope)
            app.toast?.let { Toast(it, Modifier.align(Alignment.BottomCenter)) }
        }
    }
}

/** Three text tabs with an accent dot under the current one (camera colours on Scan, solid over the river). */
@Composable
fun Nav(app: AppState, modifier: Modifier) {
    val c = LocalPond.current
    val cam = app.tab == Tab.SCAN
    val solid = app.tab == Tab.HISTORY
    Box(modifier.fillMaxWidth()) {
        if (solid) Canvas(Modifier.matchParentSize()) {  // the bar fades up into the river
            drawRect(Brush.verticalGradient(0f to Color.Transparent, .22f to c.bg, 1f to c.bg))
        }
        Row(Modifier.fillMaxWidth().navigationBarsPadding().padding(top = if (solid) 26.dp else 0.dp, bottom = 22.dp),
            horizontalArrangement = Arrangement.spacedBy(34.dp, Alignment.CenterHorizontally)) {
            for (t in Tab.entries) {
                val on = app.tab == t
                val color = when {
                    cam && on -> Cam.ink
                    cam -> Cam.sub
                    on -> c.ink
                    else -> c.sub
                }
                Column(
                    Modifier.height(44.dp).clickable(remember { MutableInteractionSource() }, null) { app.go(t) }.padding(horizontal = 6.dp),
                    horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center
                ) {
                    Text(t.name, style = Type.nav.copy(color = color))
                    Spacer(Modifier.height(6.dp))
                    Box(Modifier.size(5.dp).clip(RoundedCornerShape(50)).background(if (on) (if (cam) Cam.accent else c.accent) else Color.Transparent))
                }
            }
        }
    }
}

@Composable
fun Toast(text: String, modifier: Modifier) {
    val c = LocalPond.current
    Text(text, style = Type.toast.copy(color = c.bg, textAlign = TextAlign.Center),
        modifier = modifier.navigationBarsPadding().padding(bottom = 90.dp).widthIn(max = 320.dp).clip(RoundedCornerShape(20.dp))
            .background(c.ink).padding(horizontal = 16.dp, vertical = 10.dp))
}

/** Brand block + settings button, shared by Create and History. */
@Composable
fun Header(title: String, tag: String, onSettings: () -> Unit, modifier: Modifier = Modifier) {
    val c = LocalPond.current
    Row(modifier.fillMaxWidth().padding(start = 28.dp, end = 14.dp, top = 8.dp), verticalAlignment = Alignment.Top) {
        Column(Modifier.weight(1f).padding(top = 6.dp)) {
            Text(title, style = Type.brand.copy(color = c.ink))
            Text(tag, style = Type.tag.copy(color = c.sub), modifier = Modifier.padding(top = 2.dp))
        }
        Box(Modifier.size(44.dp).clickable { onSettings() }, contentAlignment = Alignment.Center) {
            Icon24(G.SETTINGS, c.ink, 24.dp, 1.6f)
        }
    }
}

/** "Today", "Yesterday", or "Oct 26". */
fun dayLabel(ts: Long): String {
    val day = 86_400_000L
    val tz = java.util.TimeZone.getDefault()
    val d = (ts + tz.getOffset(ts)) / day
    val now = System.currentTimeMillis()
    val n = (now + tz.getOffset(now)) / day
    return when {
        n - d <= 0 -> "Today"
        n - d == 1L -> "Yesterday"
        else -> java.text.SimpleDateFormat("MMM d", java.util.Locale.ENGLISH).format(java.util.Date(ts))
    }
}
