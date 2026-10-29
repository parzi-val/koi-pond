package app.koipond.ui

import android.graphics.Bitmap
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import app.koipond.core.KoiCodec
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** The current pond as a small thumbnail (for settings cards and the share sheet). */
@Composable
private fun rememberThumb(app: AppState): Bitmap? {
    var t by remember(app.current.id) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(app.current.id, app.pond) {
        t = app.pond?.let { withContext(Dispatchers.Default) { Bitmap.createScaledBitmap(it, 300, 300, true) } }
            ?: withContext(Dispatchers.IO) { app.store.loadThumb(app.current.id) }
    }
    return t
}

@Composable
fun SettingsScreen(app: AppState, scope: CoroutineScope, systemDark: Boolean) {
    val c = LocalPond.current
    val dark = if (app.match) systemDark else app.theme == "dark"
    val thumb = rememberThumb(app)
    var armed by remember { mutableStateOf(false) }
    LaunchedEffect(armed) { if (armed) { delay(3000); armed = false } }
    Column(Modifier.fillMaxSize().background(c.bg).clickable(remember { MutableInteractionSource() }, null) {}.statusBarsPadding()) {
        Row(Modifier.padding(start = 14.dp, end = 14.dp, top = 8.dp), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            Box(Modifier.size(44.dp).clickable { app.settingsOpen = false }, contentAlignment = Alignment.Center) { Icon24(G.BACK, c.ink, 24.dp, 1.7f) }
            Text("Settings", style = Type.brand.copy(color = c.ink))
        }
        Column(Modifier.padding(start = 28.dp, end = 28.dp, top = 30.dp)) {
            Text("APPEARANCE", style = Type.eyebrow.copy(color = c.sub), modifier = Modifier.padding(bottom = 14.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                ThemeCard("Still water", !dark, panel = Color(0xFFD6EEE1), thumb, glow = false, Modifier.weight(1f)) { app.pickTheme("light") }
                ThemeCard("Night pond", dark, panel = Color(0xFF0E2325), thumb, glow = true, Modifier.weight(1f)) { app.pickTheme("dark") }
            }
            SwitchRow("Match system setting", "Follow your phone's light or dark mode", app.match, Modifier.padding(top = 22.dp)) {
                app.setMatch(!app.match, systemDark)
            }
            Text("HISTORY", style = Type.eyebrow.copy(color = c.sub), modifier = Modifier.padding(top = 28.dp, bottom = 6.dp))
            SwitchRow("Keep history on this phone", "Released ponds flow into your river", app.keep) { app.pickKeep(!app.keep) }
            Box(Modifier.fillMaxWidth().height(1.dp).background(c.hair))
            Text(if (armed) "Tap again to clear history" else "Clear history", style = Type.row.copy(color = c.danger),
                modifier = Modifier.fillMaxWidth().clickable {
                    if (!armed) armed = true else { armed = false; app.clearHistory(); app.say(scope, "History cleared") }
                }.padding(vertical = 16.dp))
        }
    }
}

@Composable
private fun ThemeCard(name: String, on: Boolean, panel: Color, thumb: Bitmap?, glow: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val c = LocalPond.current
    Column(modifier.clickable { onClick() }, verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Box(
            Modifier.fillMaxWidth().height(150.dp).clip(RoundedCornerShape(18.dp)).background(panel)
                .border(if (on) 2.dp else 1.dp, if (on) c.ink else c.hair, RoundedCornerShape(18.dp)),
            contentAlignment = Alignment.Center
        ) {
            thumb?.let {
                val shape = RoundedCornerShape(if (glow) 12.dp else 0.dp)
                Image(it.asImageBitmap(), null, Modifier.size(100.dp)
                    .then(if (glow) Modifier.shadow(18.dp, shape, ambientColor = Color(0x40BEF0DC), spotColor = Color(0x40BEF0DC)) else Modifier)
                    .clip(shape), contentScale = ContentScale.Crop)
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Canvas(Modifier.size(18.dp)) {  // tick: filled ink circle with a check in the background colour
                if (!on) return@Canvas
                scale(size.width / 24f, pivot = Offset.Zero) {
                    drawCircle(c.ink, 11f, Offset(12f, 12f))
                    drawPath(G.path(G.TICK_MARK), c.bg, style = Stroke(2.4f, cap = StrokeCap.Round, join = StrokeJoin.Round))
                }
            }
            Text(name, style = Type.cardName.copy(color = c.ink))
        }
    }
}

@Composable
private fun SwitchRow(title: String, sub: String, on: Boolean, modifier: Modifier = Modifier, onToggle: () -> Unit) {
    val c = LocalPond.current
    Column(modifier) {
        Box(Modifier.fillMaxWidth().height(1.dp).background(c.hair))
        Row(Modifier.fillMaxWidth().padding(vertical = 16.dp), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(title, style = Type.row.copy(color = c.ink))
                Text(sub, style = Type.rowSmall.copy(color = c.sub))
            }
            val knob by animateDpAsState(if (on) 23.dp else 3.dp, tween(200), label = "knob")
            Box(Modifier.size(52.dp, 32.dp).clip(RoundedCornerShape(16.dp)).background(if (on) c.accent else c.swOff).clickable { onToggle() }) {
                Box(Modifier.offset(x = knob, y = 3.dp).size(26.dp).clip(CircleShape).background(c.knob))
            }
        }
    }
}

@Composable
fun ShareSheet(app: AppState, scope: CoroutineScope) {
    val c = LocalPond.current
    val ctx = LocalContext.current
    val thumb = rememberThumb(app)
    val file = app.store.full(app.current.id)
    Box(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxSize().background(Color(0x6B0E2325)).clickable(remember { MutableInteractionSource() }, null) { app.shareOpen = false })
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().clip(RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp)).background(c.sheet)
                .clickable(remember { MutableInteractionSource() }, null) {}.navigationBarsPadding()
                .padding(start = 28.dp, end = 28.dp, top = 14.dp, bottom = 28.dp),
            verticalArrangement = Arrangement.spacedBy(22.dp)
        ) {
            Box(Modifier.align(Alignment.CenterHorizontally).size(40.dp, 4.dp).clip(RoundedCornerShape(2.dp)).background(c.handle))
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(18.dp)) {
                Box(Modifier.size(84.dp).clip(RoundedCornerShape(14.dp)).border(1.5.dp, c.ink, RoundedCornerShape(14.dp))) {
                    thumb?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
                }
                Column(Modifier.weight(1f)) {
                    Text("Share this pond", style = Type.sheetTitle.copy(color = c.ink))
                    Text(app.current.msg, style = Type.sheetMsg.copy(color = c.sub), modifier = Modifier.padding(top = 4.dp))
                    Text("${KoiCodec.bytesOf(app.current.msg).size} bytes · every koi reads back", style = Type.small.copy(color = c.sub),
                        modifier = Modifier.padding(top = 4.dp))
                }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Act("Share…", G.SHARE, primary = true, Modifier.weight(1f)) { sharePond(ctx, file) }
                Act("Save image", G.SAVE, primary = false, Modifier.weight(1f)) {
                    scope.launch {
                        val ok = withContext(Dispatchers.IO) { runCatching { savePond(ctx, file, "koi-pond-${app.current.id}") }.isSuccess }
                        app.say(scope, if (ok) "Saved to Photos" else "Couldn't save the image")
                    }
                }
                Act("Copy", G.COPY, primary = false, Modifier.weight(1f)) { copyPondImage(ctx, file); app.say(scope, "Pond copied") }
            }
            Text("The message travels inside the image itself.", style = Type.note.copy(color = c.sub, textAlign = TextAlign.Center),
                modifier = Modifier.fillMaxWidth())
        }
    }
}

@Composable
private fun Act(label: String, d: String, primary: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val c = LocalPond.current
    Column(modifier.clickable { onClick() }, horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Box(Modifier.size(60.dp).clip(CircleShape).background(if (primary) c.accent else Color.Transparent)
            .border(1.5.dp, if (primary) c.accent else c.ink, CircleShape), contentAlignment = Alignment.Center) {
            Icon24(d, if (primary) c.onAccent else c.ink, 26.dp, 1.7f)
        }
        Text(label, style = Type.act.copy(color = c.ink))
    }
}
