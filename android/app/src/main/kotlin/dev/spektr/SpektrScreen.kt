package dev.spektr

import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
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
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.res.ResourcesCompat

/** Which picker, if any, is open. */
private enum class Sheet { None, Mode, Theme }

/**
 * The v2 screen: loading, idle (pick and start), or the grid with chrome.
 *
 * Mode and theme are pickable now. The chrome is tap-to-toggle over the grid:
 * the panel is meant to be watched, and a permanent row of controls across a
 * visualiser is the thing you notice instead of the picture.
 */
@Composable
fun SpektrScreen(onStartCapture: () -> Unit, onStopCapture: () -> Unit) {
    val engine = EngineManager.engine
    val error = EngineManager.error
    val palette = EngineManager.palette
    val capturing = CaptureController.state == CaptureController.State.Capturing

    val view = LocalView.current
    LaunchedEffect(capturing) {
        view.keepScreenOn = capturing
    }

    Surface(color = Color(palette?.bg ?: 0xFF000000.toInt()), modifier = Modifier.fillMaxSize()) {
        when {
            error != null -> ErrorView(error)
            engine == null || palette == null -> LoadingView()
            !capturing -> IdleView(engine, palette, onStartCapture)
            else -> CaptureView(engine, palette, onStopCapture)
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
private fun IdleView(engine: PyEngine, palette: Palette, onStart: () -> Unit) {
    var sheet by remember { mutableStateOf(Sheet.None) }

    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(20.dp)) {
            Text(
                "spektr",
                color = Color(palette.fg),
                fontFamily = FontFamily(Font(R.font.dejavu_sans_bold)),
                fontSize = 40.sp,
            )
            // Choosable before capture starts, not only after: picking a mode
            // should not cost a trip through the OS consent dialog.
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Chip(EngineManager.mode, palette) { sheet = Sheet.Mode }
                Chip(EngineManager.theme, palette) { sheet = Sheet.Theme }
            }
            Chip("start capture", palette, emphasis = true, onClick = onStart)
        }
    }
    Pickers(engine, palette, sheet) { sheet = it }
}

@Composable
private fun CaptureView(engine: PyEngine, palette: Palette, onStop: () -> Unit) {
    var sheet by remember { mutableStateOf(Sheet.None) }
    var chrome by remember { mutableStateOf(true) }

    Box(Modifier.fillMaxSize()) {
        GridView(palette)
        // A tap anywhere on the picture shows or hides the controls. No ripple
        // and no indication: the target is the whole visualiser, and a ripple
        // across it is a flash of grey over the thing being watched.
        Box(
            Modifier.fillMaxSize().clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
            ) { chrome = !chrome }
        )
        if (chrome) {
            Row(
                modifier = Modifier.align(Alignment.TopCenter).padding(8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Chip(EngineManager.mode, palette) { sheet = Sheet.Mode }
                Chip(EngineManager.theme, palette) { sheet = Sheet.Theme }
                Chip("stop", palette, onClick = onStop)
            }
        }
    }
    Pickers(engine, palette, sheet) { sheet = it }
}

/**
 * A control drawn in the theme's own colours.
 *
 * Material's Button brings its own palette, which on a themed visualiser means
 * purple furniture over a gruvbox grid. These take their colours from the same
 * place the picture does, so switching theme moves the whole screen.
 */
@Composable
private fun Chip(label: String, palette: Palette, emphasis: Boolean = false, onClick: () -> Unit) {
    val fg = Color(palette.fg)
    Text(
        label,
        color = if (emphasis) Color(palette.bg) else fg,
        fontSize = 14.sp,
        fontWeight = if (emphasis) FontWeight.Bold else FontWeight.Normal,
        modifier = Modifier
            .clickable(onClick = onClick)
            .background(if (emphasis) fg else Color(palette.bg).copy(alpha = 0.7f), RoundedCornerShape(6.dp))
            .border(1.dp, fg.copy(alpha = 0.45f), RoundedCornerShape(6.dp))
            .padding(horizontal = 14.dp, vertical = 8.dp),
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun Pickers(engine: PyEngine, palette: Palette, sheet: Sheet, onSheet: (Sheet) -> Unit) {
    if (sheet == Sheet.None) return
    val state = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    LaunchedEffect(sheet) {
        if (sheet == Sheet.Theme) EngineManager.loadSwatches()
    }

    ModalBottomSheet(
        onDismissRequest = { onSheet(Sheet.None) },
        sheetState = state,
        containerColor = Color(palette.bg),
        contentColor = Color(palette.fg),
    ) {
        when (sheet) {
            Sheet.Mode -> ModeList(engine.modes, palette) {
                EngineManager.selectMode(it)
                onSheet(Sheet.None)
            }
            Sheet.Theme -> ThemeList(engine.themes, EngineManager.swatches, palette) {
                EngineManager.selectTheme(it)
                onSheet(Sheet.None)
            }
            Sheet.None -> Unit
        }
    }
}

/**
 * Opens on the current entry rather than at the top.
 *
 * Fifty-odd rows means the one you are on is usually off screen, and a picker
 * that makes you hunt for where you already are is a picker you stop using.
 */
@Composable
private fun rememberListAt(index: Int) =
    rememberLazyListState(initialFirstVisibleItemIndex = (index - 3).coerceAtLeast(0))

@Composable
private fun ModeList(modes: List<String>, palette: Palette, onPick: (String) -> Unit) {
    val current = EngineManager.mode
    LazyColumn(
        state = rememberListAt(modes.indexOf(current)),
        modifier = Modifier.fillMaxWidth().heightIn(max = 520.dp),
    ) {
        items(modes, key = { it }) { name ->
            PickerRow(name, name == current, palette) { onPick(name) }
        }
    }
}

@Composable
private fun ThemeList(
    themes: List<String>,
    swatches: List<ThemeSwatch>,
    palette: Palette,
    onPick: (String) -> Unit,
) {
    val current = EngineManager.theme
    val byName = remember(swatches) { swatches.associateBy { it.name } }
    LazyColumn(
        state = rememberListAt(themes.indexOf(current)),
        modifier = Modifier.fillMaxWidth().heightIn(max = 520.dp),
    ) {
        items(themes, key = { it }) { name ->
            PickerRow(name, name == current, palette, swatch = byName[name]) { onPick(name) }
        }
    }
}

@Composable
private fun PickerRow(
    label: String,
    selected: Boolean,
    palette: Palette,
    swatch: ThemeSwatch? = null,
    onClick: () -> Unit,
) {
    val fg = Color(palette.fg)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            // Selection is a background wash, not a checkmark: the row has to
            // be findable while scrolling past at arm's length.
            .background(if (selected) fg.copy(alpha = 0.14f) else Color.Transparent)
            .padding(horizontal = 20.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text(
            label,
            color = fg,
            fontSize = 16.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
            modifier = Modifier.weight(1f),
        )
        if (swatch != null) {
            Row(
                Modifier
                    .background(Color(swatch.bg), RoundedCornerShape(4.dp))
                    .border(1.dp, fg.copy(alpha = 0.25f), RoundedCornerShape(4.dp))
                    .padding(3.dp)
            ) {
                for (c in swatch.colors) {
                    Box(Modifier.size(width = 16.dp, height = 20.dp).background(Color(c)))
                }
            }
        }
    }
}

/** Measured cell geometry: how wide/tall one grid cell is at the chosen glyph size. */
private data class CellMetrics(val w: Float, val h: Float, val fontSizeSp: Float)

/**
 * Grid font plus a fallback for what it does not have.
 *
 * DejaVu Sans covers braille, the block elements and box drawing — which is
 * almost everything the modes emit. It does not cover halfwidth katakana, and
 * Matrix is built entirely out of it: shipping the picker without this makes
 * one of the fifty-two modes forty-five glyphs of tofu, which reads as a crash
 * rather than as a missing font.
 *
 * [has] caches the per-codepoint answer because `hasGlyph` takes a String and
 * the alternative is allocating one per cell per frame.
 */
private class GridPaints(val main: android.graphics.Paint, val fallback: android.graphics.Paint) {
    private val known = HashMap<Int, Boolean>(512)

    fun paintFor(code: Int): android.graphics.Paint =
        if (known.getOrPut(code) { main.hasGlyph(String(Character.toChars(code))) }) main else fallback
}

/**
 * The Compose renderer. Measures its own cell metrics from the grid font,
 * reports the resulting cell counts to [EngineManager] (which renders at
 * exactly that size), and draws each frame run-length-encoded: one rect per
 * background run, one drawText per foreground run — the same idea
 * make_strips uses on desktop, reimplemented against the wire format.
 */
@Composable
fun GridView(palette: Palette) {
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
                "█",
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
    val paints = remember(cell) {
        val size = with(density) { (cell?.fontSizeSp ?: 12f).sp.toPx() }
        val main = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            typeface = ResourcesCompat.getFont(context, R.font.dejavu_sans)
            textSize = size
        }
        // MONOSPACE rather than a second bundled font: it resolves through the
        // platform's fallback chain, which reaches Noto Sans CJK, so the app
        // does not carry a CJK font to draw forty-five katakana.
        val fallback = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            typeface = android.graphics.Typeface.MONOSPACE
            textSize = size
        }
        GridPaints(main, fallback)
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
        drawGrid(frame, palette, c, paints)
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawGrid(
    frame: FrameBuf,
    palette: Palette,
    cell: CellMetrics,
    paints: GridPaints,
) {
    val cols = frame.w
    val rows = frame.h
    if (cols <= 0 || rows <= 0) return
    val ramp = palette.ramp
    val rampColor = { idx: Int -> if (idx < ramp.size) Color(ramp[idx]) else Color(0xFF000000) }
    val bgIndex = { i: Int -> if (frame.planes == 3) frame.bidx!![i].toInt() and 0xFF else -1 }
    // Skia draws text from the baseline; the wire format positions cells by
    // their top edge. `-top` is the ascent above the baseline for this font.
    val baseline = -paints.main.fontMetrics.top

    // Background plane first, run-length over same-coloured cells per row.
    var i = 0
    while (i < rows * cols) {
        val row = i / cols
        val col = i - row * cols
        val bg = bgIndex(i)
        val color = if (bg >= 0) rampColor(bg) else Color(palette.bg)
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
            val paint = paints.paintFor(code)
            paint.color = ramp.getOrElse(fg) { 0xFF000000.toInt() }
            drawIntoCanvas { canvas ->
                if (paint === paints.main) {
                    // Codepoints may be astral (some modes draw past U+FFFF),
                    // so the run is built with appendCodePoint, never
                    // Char.toChar.
                    val sb = StringBuilder((j - i) * 2)
                    for (k in 0 until (j - i)) sb.appendCodePoint(code)
                    canvas.nativeCanvas.drawText(
                        sb.toString(), col * cell.w, row * cell.h + baseline, paint,
                    )
                } else {
                    // A fallback glyph has its own advance width, which is not
                    // this grid's cell width — drawn as one string the run
                    // would drift out of its column. One cell at a time, each
                    // centred in its own box, stays on the grid whatever font
                    // the platform picked.
                    val s = String(Character.toChars(code))
                    val dx = (cell.w - paint.measureText(s)) * 0.5f
                    for (k in 0 until (j - i)) {
                        canvas.nativeCanvas.drawText(
                            s, (col + k) * cell.w + dx, row * cell.h + baseline, paint,
                        )
                    }
                }
            }
        }
        i = j
    }
}
