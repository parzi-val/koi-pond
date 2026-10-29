package app.koipond.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.PathParser
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import app.koipond.core.KoiCodec
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

// the design's river constants (390-unit frame)
private val RX = intArrayOf(100, 96, 104, 98, 102, 100)
private val DXS = intArrayOf(-4, 6, -6, 4, -3, 5)
private val TILT = floatArrayOf(-2.5f, 2f, -1.5f, 2.5f, -2f, 1.5f)
private val PADR = floatArrayOf(30f, 110f, 260f, 40f, 330f, 80f)
private const val Y0 = 170f
private const val STEP = 224f

@Composable
fun HistoryScreen(app: AppState, scope: CoroutineScope) {
    val c = LocalPond.current
    val items = app.history
    val n = items.size
    val yEnd = Y0 + STEP * n
    val h = maxOf(844f, yEnd + 210f)
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val u = maxWidth / 390
        Box(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            Box(Modifier.fillMaxWidth().height(u * h).statusBarsPadding()) {
                River(n, yEnd, h, Modifier.fillMaxSize())
                Header("Your river", "ponds drift downstream as they age", { app.settingsOpen = true })
                if (n == 0) Text("No ponds yet. Release one and your river starts here.", style = Type.empty.copy(color = c.sub, textAlign = TextAlign.Center),
                    modifier = Modifier.offset(x = u * 40, y = u * 260).width(u * 310))
                items.forEachIndexed { i, it ->
                    val cx = 195 + DXS[i % 6]
                    val cy = Y0 + 112 + STEP * i
                    val right = i % 2 == 0
                    Tile(app, it, Modifier.offset(x = u * (cx - 56), y = u * (cy - 56)).size(u * 112).rotate(TILT[i % 6])) { app.open(it, scope) }
                    Column(Modifier.offset(x = u * (if (right) 22 else 270), y = u * (cy - 42)).width(u * (if (right) 106 else 100))) {
                        val align = if (right) TextAlign.End else TextAlign.Start
                        Text(dayLabel(it.ts).uppercase(), style = Type.labDate.copy(color = c.sub, textAlign = align), modifier = Modifier.fillMaxWidth())
                        Text(it.msg, style = Type.labMsg.copy(color = c.ink, textAlign = align), maxLines = 3, overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.fillMaxWidth().padding(top = 3.dp))
                        Text("${KoiCodec.bytesOf(it.msg).size} bytes", style = Type.labBytes.copy(color = c.sub, textAlign = align),
                            modifier = Modifier.fillMaxWidth().padding(top = 3.dp))
                    }
                }
            }
        }
    }
}

@Composable
private fun Tile(app: AppState, item: PondItem, modifier: Modifier, onClick: () -> Unit) {
    val c = LocalPond.current
    var thumb by remember(item.id) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(item.id) { thumb = withContext(Dispatchers.IO) { app.store.loadThumb(item.id) } }
    Box(modifier.clip(RoundedCornerShape(14.dp)).border(1.5.dp, c.ink, RoundedCornerShape(14.dp)).clickable { onClick() }) {
        thumb?.let { Image(it.asImageBitmap(), "Pond: ${item.msg}", Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
    }
}

/** The river (edge, water, glint), its source ripples, lily pads and koi; fades out downstream. */
@Composable
private fun River(n: Int, yEnd: Float, h: Float, modifier: Modifier) {
    val c = LocalPond.current
    val d = remember(n) {
        if (n == 0) null else buildString {
            append("M 140 $Y0 L 195 $Y0")
            for (i in 0 until n) append(" A ${RX[i % 6]} 112 0 0 ${if (i % 2 == 0) 1 else 0} 195 ${Y0 + STEP * (i + 1)}")
            append(if ((n - 1) % 2 == 0) " C 150 $yEnd 90 ${yEnd + 16} 50 ${yEnd + 86}" else " C 240 $yEnd 300 ${yEnd + 16} 340 ${yEnd + 86}")
        }.let { PathParser().parsePathString(it).toPath() }
    }
    val pad = remember { G.path(G.PAD) }
    val koi = remember { G.path(G.KOI) }
    Canvas(modifier.graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen }) {
        val s = size.width / 390f
        scale(s, pivot = Offset.Zero) {
            drawCircle(c.edge.copy(alpha = c.edge.alpha * .35f), 24f, Offset(140f, Y0), style = Stroke(1f))
            drawCircle(c.edge.copy(alpha = c.edge.alpha * .18f), 34f, Offset(140f, Y0), style = Stroke(1f))
            if (d != null) {
                drawPath(d, c.edge, style = Stroke(37f, cap = StrokeCap.Round, join = StrokeJoin.Round))
                drawPath(d, c.river, style = Stroke(34f, cap = StrokeCap.Round, join = StrokeJoin.Round))
                drawPath(d, c.glint, style = Stroke(2f, cap = StrokeCap.Round, pathEffect = PathEffect.dashPathEffect(floatArrayOf(14f, 24f))))
                for (j in 0 until n) {
                    val cy = Y0 + 112 + STEP * j
                    val right = j % 2 == 0
                    val ex = 195f + (if (right) 1 else -1) * (RX[j % 6] - 4)
                    padAt(pad, ex, cy + (if (right) 10 else -4), PADR[j % 6], 1f, c)
                    padAt(pad, ex + (if (right) 10 else -10), cy - (if (right) 12 else -18), PADR[(j + 3) % 6], .65f, c)
                    if (j < n - 1 && j % 3 != 2) koiAt(koi, 195f, Y0 + STEP * (j + 1), !right, if (right) -6f else 5f, if (j % 2 == 1) c.koiB else c.koiA, c)
                }
            }
            // downstream fade: keep everything above yEnd-40, nothing below yEnd+50
            drawRect(Brush.verticalGradient(0f to Color.White, 1f to Color.Transparent, startY = yEnd - 40, endY = yEnd + 50),
                size = androidx.compose.ui.geometry.Size(390f, h), blendMode = BlendMode.DstIn)
        }
    }
}

private fun DrawScope.padAt(p: androidx.compose.ui.graphics.Path, x: Float, y: Float, rot: Float, sc: Float, c: PondColors) {
    translate(x, y) {
        rotate(rot, Offset.Zero) {
            scale(sc, pivot = Offset.Zero) {
                drawPath(p, c.pad, style = Fill)
                drawPath(p, c.padStroke, style = Stroke(.8f, join = StrokeJoin.Round))
            }
        }
    }
}

private fun DrawScope.koiAt(p: androidx.compose.ui.graphics.Path, x: Float, y: Float, flip: Boolean, tilt: Float, fill: Color, c: PondColors) {
    // SVG: flip ? translate(x+14, y-6) scale(-1 1) rotate(tilt 14 6) : translate(x-14, y-6) rotate(tilt 14 6)
    translate(if (flip) x + 14 else x - 14, y - 6) {
        scale(if (flip) -1f else 1f, 1f, pivot = Offset.Zero) {
            rotate(tilt, Offset(14f, 6f)) {
                drawPath(p, fill, style = Fill)
                drawPath(p, c.koiStroke, style = Stroke(.9f, join = StrokeJoin.Round))
            }
        }
    }
}
