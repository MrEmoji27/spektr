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
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.res.ResourcesCompat
import kotlin.math.ln
import kotlin.math.pow

/** Which sheet, if any, is open. */
private enum class Sheet { None, Mode, Theme, Settings, Changelog }

/**
 * The changelog, as shipped in the APK's assets.
 *
 * Read once and remembered: it is a few kilobytes, and re-reading it every
 * recomposition to draw the same text would be work for nothing.
 */
@Composable
private fun rememberChangelog(): List<String> {
    val context = LocalContext.current
    return remember {
        runCatching {
            context.assets.open("CHANGELOG.md").bufferedReader().readLines()
        }.getOrElse { listOf("The changelog did not ship with this build.") }
    }
}

/**
 * Just enough Markdown to read a changelog, and no more.
 *
 * Headings, bullets and `**bold**` are what this document is made of. A real
 * Markdown renderer would be a dependency and a lot of surface area to display
 * one file that we also write.
 */
private fun markdownLine(raw: String): AnnotatedString = buildAnnotatedString {
    var text = raw.trimEnd()
    var indent = ""
    if (text.startsWith("- ")) {
        indent = "  •  "
        text = text.removePrefix("- ")
    }
    append(indent)
    // Split on ** and alternate plain/bold; backticks become plain text, since
    // a monospace run inside a proportional paragraph reads worse than none.
    val parts = text.replace("`", "").split("**")
    parts.forEachIndexed { i, part ->
        if (i % 2 == 1) {
            withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(part) }
        } else {
            append(part)
        }
    }
}

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

    // The mode draws here too, before any consent has been given.
    //
    // The engine renders whether or not audio is arriving — silence is just
    // quiet input — so the home screen can show the mode you are about to
    // pick instead of describing it by name. Dimmed, because it is a preview
    // behind the controls and not the thing itself.
    Box(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxSize().alpha(0.35f)) { GridView(palette) }
    }
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
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Chip("‹", palette) { EngineManager.cycleMode(-1) }
                Chip(EngineManager.mode, palette) { sheet = Sheet.Mode }
                Chip("›", palette) { EngineManager.cycleMode(1) }
                Chip(EngineManager.theme, palette) { sheet = Sheet.Theme }
                Chip("⚙", palette) { sheet = Sheet.Settings }
            }
            Chip("start capture", palette, emphasis = true, onClick = onStart)
        }
        Column(
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 18.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Chip("what's new", palette) { sheet = Sheet.Changelog }
            Text(
                "made by zemo",
                color = Color(palette.fg).copy(alpha = 0.55f),
                fontSize = 12.sp,
            )
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
                Chip("‹", palette) { EngineManager.cycleMode(-1) }
                Chip(EngineManager.mode, palette) { sheet = Sheet.Mode }
                Chip("›", palette) { EngineManager.cycleMode(1) }
                Chip(EngineManager.theme, palette) { sheet = Sheet.Theme }
                Chip("⚙", palette) { sheet = Sheet.Settings }
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
            Sheet.Settings -> SettingsList(palette)
            Sheet.Changelog -> ChangelogList(palette)
            Sheet.None -> Unit
        }
    }
}

/** One `##` section of the changelog: a version, and everything said about it. */
private class Release(val title: String, val body: List<String>)

/**
 * Splits the changelog into its versions.
 *
 * Everything before the first `##` is the document's own preamble and belongs
 * to no release, so it is returned separately rather than folded into the
 * first one — collapsing the newest version should not hide the explanation
 * of what the file is.
 */
private fun releases(lines: List<String>): Pair<List<String>, List<Release>> {
    val preamble = lines.takeWhile { !it.startsWith("## ") }
    val out = mutableListOf<Release>()
    var title: String? = null
    var body = mutableListOf<String>()
    for (line in lines.drop(preamble.size)) {
        if (line.startsWith("## ")) {
            title?.let { out += Release(it, body) }
            title = line.removePrefix("## ")
            body = mutableListOf()
        } else {
            body += line
        }
    }
    title?.let { out += Release(it, body) }
    return preamble to out
}

@Composable
private fun ChangelogList(palette: Palette) {
    val lines = rememberChangelog()
    val fg = Color(palette.fg)
    val (preamble, versions) = remember(lines) { releases(lines) }
    // The newest is open and the rest are shut. What people want from a
    // changelog is what changed *this* time; the older entries are there to
    // be looked up, not to be scrolled past on the way.
    val open = remember(versions) { mutableStateListOf(*Array(versions.size) { it == 0 }) }

    LazyColumn(
        Modifier.fillMaxWidth().heightIn(max = 560.dp).padding(horizontal = 24.dp),
    ) {
        items(preamble.size) { i -> ChangelogLine(preamble[i], fg) }

        versions.forEachIndexed { v, release ->
            item {
                Row(
                    Modifier
                        .fillMaxWidth()
                        .clickable { open[v] = !open[v] }
                        .padding(top = 18.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text(if (open[v]) "▾" else "▸", color = fg, fontSize = 18.sp)
                    Text(
                        release.title,
                        color = fg,
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.weight(1f),
                    )
                    // Only when there is a count worth showing: a release
                    // written as plain paragraphs has no `###` headings, and
                    // "0 sections" reads as an empty release rather than as
                    // one that simply is not subdivided.
                    val sections = release.body.count { it.startsWith("### ") }
                    if (!open[v] && sections > 0) {
                        Text(
                            "$sections sections",
                            color = fg.copy(alpha = 0.5f),
                            fontSize = 12.sp,
                        )
                    }
                }
            }
            if (open[v]) {
                items(release.body.size) { i -> ChangelogLine(release.body[i], fg) }
            }
        }

        item { Spacer(Modifier.height(24.dp)) }
    }
}

@Composable
private fun ChangelogLine(raw: String, fg: Color) {
    when {
        raw.isBlank() -> Spacer(Modifier.height(10.dp))

        // `##` never reaches here — those are the collapsible release headers,
        // drawn by ChangelogList itself.
        raw.startsWith("### ") -> Text(
            raw.removePrefix("### "),
            color = fg,
            fontSize = 17.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(top = 14.dp, bottom = 4.dp),
        )

        raw.startsWith("# ") -> Text(
            raw.removePrefix("# "),
            color = fg,
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 8.dp),
        )

        // A table row would wrap into nonsense at this width, and the
        // changelog's tables are all "measurement | number" pairs that
        // read fine as one line. The |---|---| separator carries nothing.
        raw.startsWith("|") ->
            if (raw.replace("-", "").replace("|", "").isBlank()) {
                Spacer(Modifier.height(2.dp))
            } else {
                Text(
                    raw.trim('|', ' ').replace("|", "   ·   "),
                    color = fg.copy(alpha = 0.85f),
                    fontSize = 13.sp,
                )
            }

        else -> Text(
            markdownLine(raw),
            color = fg.copy(alpha = 0.85f),
            fontSize = 14.sp,
            lineHeight = 20.sp,
        )
    }
}

@Composable
private fun SettingsList(palette: Palette) {
    val fg = Color(palette.fg)
    Column(Modifier.fillMaxWidth().padding(bottom = 24.dp)) {
        SettingRow(
            "true black",
            "Paints the background #000000 and fades the ramp's floor into it. " +
                "An OLED pixel showing black is switched off, so on this panel that is " +
                "not a darker theme — it is less screen.",
            palette,
        ) {
            Chip(if (EngineManager.oled) "on" else "off", palette, emphasis = EngineManager.oled) {
                EngineManager.useOled(!EngineManager.oled)
            }
        }

        SettingRow(
            "detail  ${EngineManager.targetRows} rows",
            "How many rows of cells fit on the screen — the app's resolution. Modes that " +
                "draw at cell resolution rather than into braille dots (Needle's dial, VU) " +
                "look coarse at low row counts, because one cell is a very large pixel.",
            palette,
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                for (r in EngineManager.ROW_CHOICES) {
                    Chip("$r", palette, emphasis = r == EngineManager.targetRows) {
                        EngineManager.useRows(r)
                    }
                }
            }
        }

        SettingRow(
            "smooth",
            "Draws the picture instead of the glyphs. A cell is not a pixel — Chladni " +
                "computes a smooth field and then picks one half-block to stand for each " +
                "cell. This runs the mode finer and blits what it actually computed.",
            palette,
        ) {
            Chip(if (EngineManager.smooth) "on" else "off", palette, emphasis = EngineManager.smooth) {
                EngineManager.useSmooth(!EngineManager.smooth)
            }
        }

        // Log scale: the useful range is multiplicative — 0.5 and 2.0 are the
        // same size of change in opposite directions — and a linear slider
        // over 0.15..8 puts everything under 1.0 in the first eighth of it.
        val t = remember(EngineManager.sensitivity) {
            (ln(EngineManager.sensitivity / 0.15f) / ln(8f / 0.15f)).coerceIn(0f, 1f)
        }
        SettingRow(
            "sensitivity  ×%.2f".format(EngineManager.sensitivity),
            "Trim on top of the analyser's own auto-gain. Modes that trigger on level " +
                "rather than draw it — Fireworks launches at a rate set by energy — get " +
                "busier or calmer with this.",
            palette,
        ) {
            Slider(
                value = t,
                onValueChange = { EngineManager.useSensitivity(0.15f * (8f / 0.15f).pow(it)) },
                modifier = Modifier.width(280.dp),
                colors = SliderDefaults.colors(
                    thumbColor = fg,
                    activeTrackColor = fg,
                    inactiveTrackColor = fg.copy(alpha = 0.25f),
                ),
            )
        }
    }
}

@Composable
private fun SettingRow(
    title: String,
    explanation: String,
    palette: Palette,
    control: @Composable () -> Unit,
) {
    val fg = Color(palette.fg)
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, color = fg, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            Text(explanation, color = fg.copy(alpha = 0.65f), fontSize = 13.sp)
        }
        control()
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

/** How to draw one codepoint: which Paint, the string, and the tracking that puts it on the grid. */
private class Glyph(val text: String, val paint: android.graphics.Paint, val spacingEm: Float)

/**
 * Grid font, a fallback for what it does not have, and one cell of tracking.
 *
 * Two separate problems, both invisible from the Python side.
 *
 * **Coverage.** DejaVu Sans has braille, the block elements and box drawing —
 * nearly everything the modes emit — but not halfwidth katakana, and Matrix is
 * built entirely out of it. A custom Typeface carries no system fallback, so
 * those cells draw as tofu. Anything DejaVu lacks goes to a MONOSPACE Paint,
 * which resolves through the platform chain and reaches Noto Sans CJK.
 *
 * **Advance.** This is what made a dozen modes look broken. The renderer
 * measures one cell from U+2588 and draws a whole run as a single string —
 * which only lands on the grid if every glyph advances by exactly one cell,
 * and in DejaVu almost none of them do. Braille is 0.7324 em against the full
 * block's 0.7690: every braille glyph in a run lands 4.8% of a cell to the
 * left of the one before it, so a hundred-cell row finishes five cells adrift
 * and the picture shears. Every braille mode was affected — Locket, Tunnel,
 * Dither Storm, Gonio, Valentine — and the block-element modes were not, which
 * is why Kaleidoscope looked perfect in v1 and hid the bug.
 *
 * The fix is tracking, not scaling: `letterSpacing` pads each glyph's advance
 * out to exactly one cell without touching its shape, and Minikin splits that
 * padding either side, so the glyph also ends up centred in its box. One
 * drawText per run still, and the run stays on the grid however wide the
 * glyph is.
 */
private class GridPaints(
    private val main: android.graphics.Paint,
    private val fallback: android.graphics.Paint,
    private val cellW: Float,
) {
    /** Skia draws text from the baseline; the wire format positions cells by their top edge. */
    val baseline: Float = -main.fontMetrics.top

    // Cached per codepoint: hasGlyph and measureText both allocate a String,
    // and the alternative is doing that per cell per frame.
    private val known = HashMap<Int, Glyph>(512)

    // One builder for the whole frame rather than one per run.
    private val scratch = StringBuilder(256)

    /**
     * A run of [n] copies of [glyph], built into the shared scratch.
     *
     * appendCodePoint would be wrong here even though the glyph is one
     * codepoint: [Glyph.text] already holds the surrogate pair for the astral
     * codepoints some modes emit, so appending the string is both correct and
     * cheaper than re-encoding it.
     */
    fun run(glyph: Glyph, n: Int): String {
        scratch.setLength(0)
        for (k in 0 until n) scratch.append(glyph.text)
        return scratch.toString()
    }

    fun glyphFor(code: Int): Glyph = known.getOrPut(code) {
        val text = String(Character.toChars(code))
        val paint = if (main.hasGlyph(text)) main else fallback
        // Measured with tracking off, or the measurement includes the tracking
        // left over from whichever run was drawn last.
        paint.letterSpacing = 0f
        val advance = paint.measureText(text)
        Glyph(text, paint, if (advance > 0f) (cellW - advance) / paint.textSize else 0f)
    }
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

    // Glyph size derives from the view height and the chosen row count; the
    // cell metrics are the measured glyph box. U+2588 (full block) is
    // measured: it is the widest glyph the modes emit, so nothing ever
    // overflows a cell.
    val rows = EngineManager.targetRows
    val cell = remember(viewH, density, rows) {
        if (viewH <= 0) null
        else {
            val px = (viewH / rows.toFloat()).coerceIn(6f, 72f)
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
        GridPaints(main, fallback, cell?.w ?: size)
    }

    val gridW = if (cell != null) (viewW / cell.w).toInt().coerceIn(8, 400) else 0
    val gridH = if (cell != null) (viewH / cell.h).toInt().coerceIn(8, 200) else 0

    LaunchedEffect(gridW, gridH) {
        EngineManager.setGrid(gridW, gridH)
    }

    // One bitmap and one pixel buffer for the whole session, resized only when
    // the field's shape changes. Allocating a 470x270 Bitmap per frame at 30
    // fps is 130k ints of garbage a frame, and the collector notices.
    val blitter = remember { FieldBlitter() }

    Canvas(
        Modifier.fillMaxSize().onSizeChanged { size ->
            viewW = size.width
            viewH = size.height
        }
    ) {
        val frame = EngineManager.lastFrame ?: return@Canvas
        if (frame.isField) {
            blitter.draw(this, frame, palette)
            return@Canvas
        }
        val c = cell ?: return@Canvas
        drawGrid(frame, palette, c, paints)
    }
}

/**
 * Draws a field frame as one filtered bitmap.
 *
 * The picture arrives at whatever resolution the mode's own geometry has —
 * for Chladni's half-blocks that is two rows per cell, and rendered at 4x the
 * grid it is 472x272 — and is stretched to the view. Bilinear, deliberately:
 * the nodal lines of a Chladni figure are smooth curves, and the whole
 * complaint about the glyph renderer was that it drew them as staircases.
 */
private class FieldBlitter {
    private var bitmap: android.graphics.Bitmap? = null
    private var pixels = IntArray(0)
    private var lut = IntArray(0)
    private var lutFor: Palette? = null

    fun draw(
        scope: androidx.compose.ui.graphics.drawscope.DrawScope,
        frame: FrameBuf,
        palette: Palette,
    ) {
        val n = frame.size
        if (n <= 0) return
        var bmp = bitmap
        if (bmp == null || bmp.width != frame.w || bmp.height != frame.h) {
            bmp?.recycle()
            bmp = android.graphics.Bitmap.createBitmap(
                frame.w, frame.h, android.graphics.Bitmap.Config.ARGB_8888
            )
            bitmap = bmp
            pixels = IntArray(n)
        }
        if (lutFor != palette) {
            // 256 entries so the byte index needs no bounds check per pixel,
            // with everything past the ramp — FIELD_EMPTY included — reading
            // as the background.
            lut = IntArray(256) { palette.bg }
            for (i in palette.ramp.indices) lut[i] = palette.ramp[i]
            lutFor = palette
        }
        val src = frame.cidx
        val dst = pixels
        val table = lut
        for (i in 0 until n) dst[i] = table[src[i].toInt() and 0xFF]
        bmp.setPixels(dst, 0, frame.w, 0, 0, frame.w, frame.h)

        scope.drawIntoCanvas { canvas ->
            val paint = android.graphics.Paint().apply {
                isFilterBitmap = true
                isDither = true
            }
            val dstRect = android.graphics.Rect(
                0, 0, scope.size.width.toInt(), scope.size.height.toInt()
            )
            canvas.nativeCanvas.drawBitmap(bmp, null, dstRect, paint)
        }
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
    val baseline = paints.baseline

    // Background plane first, run-length over same-coloured cells per row.
    //
    // Only when there is one. A two-plane mode has no per-cell background, so
    // every rect this pass drew was the surface's own colour painted over
    // itself — one wasted rect per row, every frame, for nothing visible.
    var i = 0
    if (frame.planes == 3) {
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
            val glyph = paints.glyphFor(code)
            val paint = glyph.paint
            paint.color = ramp.getOrElse(fg) { 0xFF000000.toInt() }
            // Tracking is set per run, not once: two codepoints sharing a
            // Paint want different padding, and whichever ran last would
            // otherwise decide it.
            paint.letterSpacing = glyph.spacingEm
            // The run is one glyph repeated. A run of one — much the commonest
            // case on a mode with fine detail, where a whole row can be
            // hundreds of one-cell runs — draws the cached string directly;
            // building a StringBuilder to hold a single character was
            // thousands of allocations a frame for nothing.
            val text = if (j - i == 1) glyph.text else paints.run(glyph, j - i)
            drawIntoCanvas { canvas ->
                canvas.nativeCanvas.drawText(
                    text, col * cell.w, row * cell.h + baseline, paint,
                )
            }
        }
        i = j
    }
}
