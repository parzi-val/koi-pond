package app.koipond.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import app.koipond.core.KoiCodec
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch

@Composable
fun CreateScreen(app: AppState, scope: CoroutineScope) {
    val c = LocalPond.current
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val w = maxWidth
        val unit = w / 390   // the design's frame is 390 wide
        if (c.dark) Canvas(Modifier.fillMaxSize()) {  // Night pond: faint concentric rings around the pond
            val centre = Offset(size.width / 2, (unit * 323).toPx())
            for (r in intArrayOf(228, 290, 360)) drawCircle(Color(0x12BEF0DC), (unit * r).toPx(), centre, style = Stroke(1.dp.toPx()))
        }
        Column(Modifier.fillMaxSize().statusBarsPadding().imePadding().verticalScroll(rememberScrollState())) {
            Header("Koi Pond", "hide a message in the scales", { app.settingsOpen = true })
            Spacer(Modifier.height(28.dp))
            PondView(app, w, c.dark)
            Form(app, scope)
            Spacer(Modifier.height(110.dp))  // room for the tab bar
        }
    }
}

@Composable
private fun PondView(app: AppState, width: androidx.compose.ui.unit.Dp, dark: Boolean) {
    val c = LocalPond.current
    val size = if (dark) width * 350 / 390 else width
    val radius = if (dark) 20.dp else 0.dp
    val dim by animateFloatAsState(if (app.dirty || app.phase == Phase.DRAWING) .5f else 1f, tween(300), label = "dim")
    val drop = remember { Animatable(1f) }
    val ripple = remember { Animatable(1f) }
    LaunchedEffect(app.drops) {
        if (app.drops == 0) return@LaunchedEffect
        launch { drop.snapTo(0f); drop.animateTo(1f, tween(700, easing = LinearOutSlowInEasing)) }
        launch { ripple.snapTo(0f); ripple.animateTo(1f, tween(900, easing = FastOutSlowInEasing)) }
    }
    Box(Modifier.fillMaxWidth().height(width), contentAlignment = Alignment.Center) {
        if (dark) Canvas(Modifier.size(size + 140.dp)) {  // glow: 0 0 90px rgba(190,240,220,.22)
            drawCircle(Brush.radialGradient(listOf(Color(0x38BEF0DC), Color.Transparent)), this.size.minDimension / 2)
        }
        Box(
            Modifier.size(size).graphicsLayer { alpha = dim * (.4f + .6f * drop.value); scaleX = .985f + .015f * drop.value; scaleY = scaleX }
                .clip(RoundedCornerShape(radius))
        ) {
            app.pond?.let { Image(it.asImageBitmap(), "Your koi pond: six koi with the message hidden in their scales", Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
        }
        if (ripple.value < 1f) Canvas(Modifier.size(80.dp)) {  // ring: scale .3 -> 4.4, opacity .8 -> 0
            val s = .3f + 4.1f * ripple.value
            drawCircle(c.accent.copy(alpha = .8f * (1 - ripple.value)), this.size.minDimension / 2 * s, style = Stroke(2.dp.toPx()))
        }
    }
}

@Composable
private fun Form(app: AppState, scope: CoroutineScope) {
    val c = LocalPond.current
    val b = app.bytes
    val over = b > KoiCodec.MAX_BYTES
    val busy = app.phase == Phase.DRAWING || app.phase == Phase.CHECKING
    val canRelease = app.msg.isNotBlank() && !over && !busy
    Column(Modifier.padding(start = 28.dp, end = 28.dp, top = 28.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.Bottom) {
            Text("YOUR MESSAGE", style = Type.eyebrow.copy(color = c.sub))
            Spacer(Modifier.weight(1f))
            Text("$b of ${KoiCodec.MAX_BYTES} bytes", style = Type.eyebrow.copy(letterSpacing = Type.small.letterSpacing,
                color = if (over) c.danger else c.sub, fontWeight = if (over) FontWeight(500) else FontWeight(400)))
        }
        Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Box(Modifier.weight(1f)) {
                BasicTextField(
                    value = app.msg,
                    onValueChange = { app.msg = it },
                    textStyle = Type.input.copy(color = c.ink),
                    singleLine = true,
                    cursorBrush = SolidColor(c.accent),
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Sentences, imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(onDone = { app.release(scope) }),
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp, bottom = 10.dp),
                    decorationBox = { inner ->
                        if (app.msg.isEmpty()) Text("Write something to hide", style = Type.input.copy(color = c.sub.copy(alpha = .8f)))
                        inner()
                    }
                )
                Box(Modifier.align(Alignment.BottomStart).fillMaxWidth().height(1.dp).background(c.line))
            }
            Box(
                Modifier.size(56.dp).clip(CircleShape).background(c.accent).alpha(if (canRelease) 1f else .4f)
                    .clickable(enabled = canRelease) { app.release(scope) },
                contentAlignment = Alignment.Center
            ) { Icon24(G.RELEASE, c.onAccent, 26.dp, 1.7f, fill = G.RELEASE_EYE) }
        }
        Box(Modifier.fillMaxWidth().height(2.dp).clip(RoundedCornerShape(2.dp)).background(c.track)) {
            Box(Modifier.fillMaxWidth(minOf(1f, b / KoiCodec.MAX_BYTES.toFloat())).fillMaxHeight().clip(RoundedCornerShape(2.dp)).background(c.accent))
        }
        Row(Modifier.fillMaxWidth().heightIn(min = 44.dp), verticalAlignment = Alignment.CenterVertically) {
            Status(app)
            Spacer(Modifier.weight(1f))
            Row(Modifier.height(44.dp).clickable(enabled = !busy) { app.shareOpen = true }.padding(horizontal = 8.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Icon24(G.SHARE, c.ink, 18.dp, 1.7f)
                Spacer(Modifier.width(8.dp))
                Text("SHARE", style = Type.caps.copy(color = c.ink))
            }
        }
    }
}

@Composable
private fun Status(app: AppState) {
    val c = LocalPond.current
    val (text, ok) = when {
        app.phase == Phase.DRAWING -> "Releasing the koi…" to false
        app.phase == Phase.CHECKING -> "Reading the koi back…" to false
        app.dirty -> "Release to update the pond" to false
        app.phase == Phase.FAILED -> "A koi slipped away — release again" to false
        else -> "Every koi reads back" to true
    }
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        if (ok) Icon24(G.CHECK, c.ok, 14.dp, 2.4f) else Box(Modifier.size(7.dp).clip(CircleShape).background(c.accent))
        Text(text, style = Type.small.copy(color = c.sub))
    }
}
