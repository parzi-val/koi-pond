package app.koipond.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.vector.PathParser
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/** The design's icons, path data copied from its SVGs (24-unit viewBox unless noted). */
object G {
    const val SETTINGS = "M4 7h10M18 7h2M4 17h2M10 17h10M14 7a2 2 0 1 0 4 0a2 2 0 1 0 -4 0M6 17a2 2 0 1 0 4 0a2 2 0 1 0 -4 0"
    const val RELEASE = "M3 12c3-5 9-6 13-2l5-3v10l-5-3c-4 4-10 3-13-2z"
    const val RELEASE_EYE = "M6.5 11a.7 .7 0 1 0 1.4 0a.7 .7 0 1 0 -1.4 0"
    const val SHARE = "M12 15V4M8 8l4-4 4 4M5 12v7h14v-7"
    const val CLOSE = "M6 6l12 12M18 6L6 18"
    const val FLASH = "M13 3L5 13h6l-1 8 8-10h-6z"
    const val CHECK = "M4 12.5l5 5L20 6.5"
    const val COPY = "M10 8h7a2 2 0 0 1 2 2v7a2 2 0 0 1 -2 2h-7a2 2 0 0 1 -2 -2v-7a2 2 0 0 1 2 -2zM5 15V6a1 1 0 0 1 1-1h9"
    const val BACK = "M15 5l-7 7 7 7"
    const val SAVE = "M12 4v11M8 11l4 4 4-4M5 19h14"
    const val TICK_MARK = "M7 12.5l3.5 3.5L17 9"
    const val FISH = "M0.5 6C5 1 13 0 20 5L27.5 1.5V10.5L20 7C13 12 5 11 0.5 6Z"            // 28 x 12
    const val PAD = "M0 0 L8.46 -3.08 A9 9 0 1 1 5.79 -6.89 Z"                              // river lily pad
    const val KOI = "M0 6 C5 1 13 0 20 5 L28 1 L28 11 L20 7 C13 12 5 11 0 6 Z"               // river koi, 28 x 12

    private val cache = HashMap<String, Path>()
    fun path(d: String): Path = cache.getOrPut(d) { PathParser().parsePathString(d).toPath() }
}

/** A stroked icon on a 24-unit grid (optionally with a filled sub-path). */
@Composable
fun Icon24(d: String, color: Color, size: Dp, stroke: Float, modifier: Modifier = Modifier, fill: String? = null) {
    val p = remember(d) { G.path(d) }
    val f = remember(fill) { fill?.let { G.path(it) } }
    Canvas(modifier.size(size)) {
        scale(this.size.width / 24f, pivot = androidx.compose.ui.geometry.Offset.Zero) {
            drawPath(p, color, style = Stroke(stroke, cap = StrokeCap.Round, join = StrokeJoin.Round))
            if (f != null) drawPath(f, color, style = Fill)
        }
    }
}

/** The 28x12 fish from the scan progress row. */
@Composable
fun FishGlyph(on: Boolean, stroke: Color, fill: Color, modifier: Modifier = Modifier) {
    val p = remember { G.path(G.FISH) }
    Canvas(modifier.size(30.dp, 14.dp)) {
        scale(this.size.width / 28f, this.size.height / 12f, pivot = androidx.compose.ui.geometry.Offset.Zero) {
            if (on) drawPath(p, fill, style = Fill)
            drawPath(p, if (on) fill else stroke, style = Stroke(1f, join = StrokeJoin.Round))
        }
    }
}
