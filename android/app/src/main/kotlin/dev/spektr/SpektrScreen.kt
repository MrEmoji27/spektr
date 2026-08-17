package dev.spektr

import android.view.View
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.res.ResourcesCompat

/**
 * The v1 screen: loading, idle (start button), or the grid.
 *
 * One hardcoded mode and theme — no pickers. Chrome stays visible; auto-hide
 * and the ambient behaviour are v3.
 */
@Composable
fun SpektrScreen(onStartCapture: () -> Unit, onStopCapture: () -> Unit) {
    val engine = EngineManager.engine
    val error = EngineManager.error
    val capturing = CaptureController.state == CaptureController.State.Capturing

    val bg = engine?.bgColor ?: 0xFF000000.toInt()

    val view = LocalView.current
    LaunchedEffect(capturing) {
        view.keepScreenOn = capturing
    }

    Surface(color = Color(bg), modifier = Modifier.fillMaxSize()) {
        when {
            error != null -> ErrorView(error)
            engine == null -> LoadingView()
            !capturing -> IdleView(engine, onStartCapture)
            else -> CaptureView(engine, onStopCapture)
        }
    }
}

@Composable
private fun LoadingView() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
            CircularProgressIndicator(color = Color(0xFFb8bb26))
            Text("starting python engine…", color = Color(0xFFebdbb2), fontSize = 14.sp)
        }
    }
}

@Composable
private fun ErrorView(message: String) {
    Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("engine failed to start", color = Color(0xFFfb4934), fontSize = 16.sp)
            Text(message, color = Color(0xFFebdbb2), fontSize = 13.sp)
        }
    }
}

@Composable
private fun IdleView(engine: PyEngine, onStart: () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(20.dp)) {
            Text(
                "spektr",
                color = Color(engine.fgColor),
                fontFamily = FontFamily(Font(R.font.dejavu_sans_bold)),
                fontSize = 40.sp,
            )
            Text(
                "${EngineManager.MODE} — ${engine.theme}",
                color = Color(engine.fgColor),
                fontSize = 14.sp,
            )
            Button(onClick = onStart) {
                Text("start capture", color = Color(0xFF1d2021))
            }
        }
    }
}

@Composable
private fun CaptureView(engine: PyEngine, onStop: () -> Unit) {
    Box(Modifier.fillMaxSize()) {
        GridView(engine)
        Text(
            "${EngineManager.MODE} — ${engine.theme}",
            color = Color(engine.fgColor),
            fontSize = 11.sp,
            modifier = Modifier.align(Alignment.TopCenter).padding(6.dp),
        )
        Button(
            onClick = onStop,
            modifier = Modifier.align(Alignment.TopEnd).padding(6.dp),
        ) {
            Text("stop", color = Color(0xFF1d2021))
        }
    }
}

/** Measured cell geometry: how wide/tall one grid cell is at the chosen glyph size. */
private data class CellMetrics(val w: Float, val h: Float, val fontSizeSp: Float)

/**
 * The Compose renderer. Measures its own cell metrics from the grid font,
 * reports the resulting cell counts to [EngineManager] (which renders at
 * exactly that size), and draws each frame run-length-encoded: one rect per
 * background run, one drawText per foreground run — the same idea
 * make_strips uses on desktop, reimplemented against the wire format.
 */
@Composable
fun GridView(engine: PyEngine) {
    val fontFamily = FontFamily(Font(R.font.dejavu_sans))
    val textMeasurer = rememberTextMeasurer()
    val density = LocalDensity.current

    var viewW by remember { mutableStateOf(0) }
    var viewH by remember { mutableStateOf(0) }

    // Glyph size derives from the view height (~40 rows on a tablet); the cell
    // metrics are the measured glyph box. U+2588 (full block) is measured: it
    // is the widest glyph the modes emit, so nothing ever overflows a cell.
    val cell = remember(viewH, density) {
        if (viewH <= 0) null
        else {
            val px = (viewH / 40f).coerceIn(8f, 72f)
            val sp = with(density) { px.toSp() }
            val layout = textMeasurer.measure(
                "\u2588",
                style = TextStyle(fontSize = sp, fontFamily = fontFamily),
            )
            CellMetrics(layout.size.width.toFloat(), layout.size.height.toFloat(), sp.value)
        }
    }

    // One Paint for the whole grid, not a TextStyle per run.
    //
    // The first version called Compose's drawText once per run with a freshly
    // built TextStyle. Every one of those is a full text layout — measure and
    // shape — and the new style object each time meant nothing could be cached
    // between them. On the tablet that was 93 ms a frame at the 50th
    // percentile against a 33 ms target, with the GPU idle at 5 ms: all of it
    // was text layout on the CPU.
    //
    // A grid renderer does not need layout. Every cell is one glyph in a known
    // box, so this is a direct Skia draw with a reusable Paint — set the
    // colour, draw the run, move on.
    val context = LocalContext.current
    val paint = remember(cell, fontFamily) {
        android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            typeface = ResourcesCompat.getFont(context, R.font.dejavu_sans)
            textSize = with(density) { (cell?.fontSizeSp ?: 12f).sp.toPx() }
        }
    }

    val gridW = if (cell != null) (viewW / cell.w).toInt().coerceIn(8, 400) else 0
    val gridH = if (cell != null) (viewH / cell.h).toInt().coerceIn(8, 200) else 0

    LaunchedEffect(gridW, gridH) {
        EngineManager.setGrid(gridW, gridH)
    }

    Canvas(
        Modifier.fillMaxSize().onSizeChanged { size ->
            viewW = size.width
            viewH = size.height
        }
    ) {
        val frame = EngineManager.lastFrame ?: return@Canvas
        val c = cell ?: return@Canvas
        drawGrid(frame, engine, c, paint)
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawGrid(
    frame: FrameBuf,
    engine: PyEngine,
    cell: CellMetrics,
    paint: android.graphics.Paint,
) {
    val cols = frame.w
    val rows = frame.h
    if (cols <= 0 || rows <= 0) return
    val ramp = engine.ramp
    val rampColor = { idx: Int -> if (idx < ramp.size) Color(ramp[idx]) else Color(0xFF000000) }
    val bgIndex = { i: Int -> if (frame.planes == 3) frame.bidx!![i].toInt() and 0xFF else -1 }
    // Skia draws text from the baseline; the wire format positions cells by
    // their top edge. `-top` is the ascent above the baseline for this font.
    val baseline = -paint.fontMetrics.top

    // Background plane first, run-length over same-coloured cells per row.
    var i = 0
    while (i < rows * cols) {
        val row = i / cols
        val col = i - row * cols
        val bg = bgIndex(i)
        val color = if (bg >= 0) rampColor(bg) else Color(engine.bgColor)
        var j = i + 1
        val rowEnd = (row + 1) * cols
        while (j < rowEnd && bgIndex(j) == bg) j++
        drawRect(
            color,
            topLeft = Offset(col * cell.w, row * cell.h),
            size = Size((j - i) * cell.w, cell.h),
        )
        i = j
    }

    // Foreground plane: runs of same glyph + same colour, one drawText each.
    i = 0
    while (i < rows * cols) {
        val row = i / cols
        val col = i - row * cols
        val code = frame.codes[i]
        val fg = frame.cidx[i].toInt() and 0xFF
        var j = i + 1
        val rowEnd = (row + 1) * cols
        while (j < rowEnd && frame.codes[j] == code && (frame.cidx[j].toInt() and 0xFF) == fg) j++
        if (code != 0) {
            // Codepoints may be astral (some modes draw past U+FFFF), so the
            // run is built with appendCodePoint, never Char.toChar.
            val sb = StringBuilder((j - i) * 2)
            for (k in 0 until (j - i)) sb.appendCodePoint(code)
            paint.color = ramp.getOrElse(fg) { 0xFF000000.toInt() }
            drawIntoCanvas { canvas ->
                canvas.nativeCanvas.drawText(
                    sb.toString(),
                    col * cell.w,
                    row * cell.h + baseline,
                    paint,
                )
            }
        }
        i = j
    }
}
