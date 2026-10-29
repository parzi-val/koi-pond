@file:OptIn(androidx.compose.ui.text.ExperimentalTextApi::class)

package app.koipond.ui

import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import app.koipond.R

/** Tokens from the design (Koi Pond.html): "Still water" (light) and "Night pond" (dark). */
@Immutable
data class PondColors(
    val dark: Boolean,
    val bg: Color, val ink: Color, val sub: Color, val line: Color, val track: Color, val hair: Color,
    val ok: Color, val accent: Color, val onAccent: Color, val sheet: Color, val handle: Color, val danger: Color,
    val swOff: Color, val knob: Color, val river: Color, val edge: Color, val glint: Color, val pad: Color,
    val padStroke: Color, val koiA: Color, val koiB: Color, val koiStroke: Color,
)

val StillWater = PondColors(
    dark = false,
    bg = Color(0xFFD6EEE1), ink = Color(0xFF17353A), sub = Color(0xFF3D6168), line = Color(0x6617353A), track = Color(0x2417353A),
    hair = Color(0x3817353A), ok = Color(0xFF2F6B4A), accent = Color(0xFFE8552F), onAccent = Color.White, sheet = Color(0xFFEEF8F2),
    handle = Color(0x4717353A), danger = Color(0xFFB3361B), swOff = Color(0x7317353A), knob = Color.White,
    river = Color(0xFFA7DCD3), edge = Color(0xFF17353A), glint = Color(0x99FFFFFF), pad = Color(0xFF8FCF88), padStroke = Color(0xFF17353A),
    koiA = Color(0xFFE8552F), koiB = Color(0xFFF2C14E), koiStroke = Color(0xFF17353A),
)

val NightPond = PondColors(
    dark = true,
    bg = Color(0xFF0E2325), ink = Color(0xFFEAF4EE), sub = Color(0xFFA9C4BF), line = Color(0x59EAF4EE), track = Color(0x29EAF4EE),
    hair = Color(0x33EAF4EE), ok = Color(0xFF7FD1A3), accent = Color(0xFFFF7A4D), onAccent = Color(0xFF0E2325), sheet = Color(0xFF14302F),
    handle = Color(0x4DEAF4EE), danger = Color(0xFFFF9B85), swOff = Color(0x66EAF4EE), knob = Color(0xFF0E2325),
    river = Color(0xFF15464A), edge = Color(0x80BEF0DC), glint = Color(0x47BEF0DC), pad = Color(0xFF3F8F62), padStroke = Color(0x80BEF0DC),
    koiA = Color(0xFFFF7A4D), koiB = Color(0xFFF2C14E), koiStroke = Color(0xFF0E2325),
)

/** The camera view is always dark, whatever the theme. */
object Cam {
    val bg = Color(0xFF16221F)
    val ink = Color(0xFFEAF4EE)
    val sub = Color(0xFFA9C4BF)
    val glass = Color(0x990E2325)
    val accent = Color(0xFFFF7A4D)
    val bracket = Color(0xFFD6EEE1)
    val found = Color(0xFF7FD1A3)
}

val LocalPond = staticCompositionLocalOf { StillWater }

private fun nr(w: Int, style: FontStyle = FontStyle.Normal) = Font(
    if (style == FontStyle.Italic) R.font.newsreader_italic else R.font.newsreader, FontWeight(w), style,
    variationSettings = FontVariation.Settings(FontVariation.weight(w))
)
private fun inter(w: Int) = Font(R.font.inter, FontWeight(w), variationSettings = FontVariation.Settings(FontVariation.weight(w)))

val Newsreader = FontFamily(nr(400), nr(500), nr(400, FontStyle.Italic))
val Inter = FontFamily(inter(400), inter(500))

/** Type styles, sized as in the design's 390-pt frame. */
object Type {
    val brand = TextStyle(fontFamily = Newsreader, fontWeight = FontWeight(500), fontSize = 26.sp, lineHeight = 28.6.sp, letterSpacing = (-.01).em)
    val tag = TextStyle(fontFamily = Newsreader, fontStyle = FontStyle.Italic, fontSize = 15.sp, lineHeight = 19.5.sp)
    val eyebrow = TextStyle(fontFamily = Inter, fontSize = 11.sp, letterSpacing = .14.em)
    val small = TextStyle(fontFamily = Inter, fontSize = 12.sp)
    val nav = TextStyle(fontFamily = Inter, fontWeight = FontWeight(500), fontSize = 12.sp, letterSpacing = .16.em)
    val caps = TextStyle(fontFamily = Inter, fontWeight = FontWeight(500), fontSize = 12.sp, letterSpacing = .14.em)
    val input = TextStyle(fontFamily = Newsreader, fontSize = 24.sp)
    val found = TextStyle(fontFamily = Newsreader, fontWeight = FontWeight(500), fontSize = 32.sp, lineHeight = 35.2.sp, letterSpacing = (-.01).em)
    val sheetTitle = TextStyle(fontFamily = Newsreader, fontWeight = FontWeight(500), fontSize = 24.sp, lineHeight = 26.4.sp, letterSpacing = (-.01).em)
    val sheetMsg = TextStyle(fontFamily = Newsreader, fontStyle = FontStyle.Italic, fontSize = 17.sp)
    val pill = TextStyle(fontFamily = Newsreader, fontSize = 19.sp)
    val row = TextStyle(fontFamily = Newsreader, fontSize = 19.sp)
    val rowSmall = TextStyle(fontFamily = Inter, fontSize = 13.sp)
    val cardName = TextStyle(fontFamily = Newsreader, fontSize = 18.sp)
    val act = TextStyle(fontFamily = Inter, fontWeight = FontWeight(500), fontSize = 12.sp, letterSpacing = .08.em)
    val note = TextStyle(fontFamily = Newsreader, fontStyle = FontStyle.Italic, fontSize = 15.sp)
    val reading = TextStyle(fontFamily = Newsreader, fontStyle = FontStyle.Italic, fontSize = 24.sp)
    val hint = TextStyle(fontFamily = Inter, fontSize = 13.sp)
    val toast = TextStyle(fontFamily = Inter, fontWeight = FontWeight(500), fontSize = 13.sp, lineHeight = 17.5.sp)
    val labDate = TextStyle(fontFamily = Inter, fontSize = 10.5.sp, letterSpacing = .12.em)
    val labMsg = TextStyle(fontFamily = Newsreader, fontSize = 18.sp, lineHeight = 20.7.sp)
    val labBytes = TextStyle(fontFamily = Inter, fontSize = 11.sp)
    val empty = TextStyle(fontFamily = Newsreader, fontStyle = FontStyle.Italic, fontSize = 20.sp, lineHeight = 27.sp)
}
