package dev.spektr

import android.content.Context
import android.content.SharedPreferences
import android.content.pm.ApplicationInfo
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext

/**
 * App-scoped owner of the Python engine and the frame pipeline.
 *
 * One thread ("spektr-py") carries every Python call: construction, pushes
 * from the capture thread, the theme switch and one render per frame tick.
 * That serialises the engine's only shared mutable state (its ring buffer,
 * palette and mode scratch) without locks, and it keeps the JNI crossing off
 * the main thread.
 *
 * The render loop is paced at ~30 fps — the design's target rate, half the
 * desktop's, because every frame costs a JNI crossing and the panel is
 * watched from across a room. Frames are produced even while the user has not
 * consented yet, so the first post-consent picture is not a blank grid.
 */
object EngineManager {

    /** What a fresh install starts on. Kaleidoscope is a half-block mode, so it exercises the background plane on day one. */
    const val DEFAULT_MODE = "Kaleidoscope"
    const val DEFAULT_THEME = "gruvbox"
    private const val FRAME_MS = 33L

    private const val PREFS = "spektr"
    private const val KEY_MODE = "mode"
    private const val KEY_THEME = "theme"
    private const val KEY_OLED = "oled"
    private const val KEY_SENSITIVITY = "sensitivity"
    private const val KEY_SMOOTH = "smooth"
    private const val KEY_ROWS = "rows"

    private val pyContext = newSingleThreadContext("spektr-py")
    private val scope = CoroutineScope(pyContext + SupervisorJob())

    var engine: PyEngine? by mutableStateOf(null)
        private set
    var error: String? by mutableStateOf(null)
        private set

    /** The newest parsed frame, drawn by [GridView]. */
    var lastFrame: FrameBuf? by mutableStateOf(null)
        private set

    /**
     * The selected mode. Read by the render loop every tick and by the chrome,
     * so it is Compose state rather than a plain field.
     *
     * Switching is a pure assignment: Python takes the mode name per render
     * call and drops the previous mode's scratch itself. Nothing has to be
     * torn down, so a switch cannot half-apply and cannot race a frame.
     */
    var mode by mutableStateOf(DEFAULT_MODE)
        private set

    var theme by mutableStateOf(DEFAULT_THEME)
        private set

    /** Colours of the current theme; null until the engine has loaded one. */
    var palette: Palette? by mutableStateOf(null)
        private set

    /**
     * True black. An OLED pixel showing #000000 is off, so this is the
     * difference between a dark theme and a dark *screen* — and on this panel
     * it is most of the picture.
     */
    var oled by mutableStateOf(false)
        private set

    /** Manual trim on the analyser, 0.15..8. Desktop's `[` and `]`. */
    var sensitivity by mutableStateOf(1.0f)
        private set

    /**
     * Draw the picture rather than the glyphs.
     *
     * A cell is not a pixel: Chladni computes a smooth nodal field and then
     * picks one half-block to stand for each cell, which on this screen means
     * a 118x34 mosaic of a thing that had far more detail in it. With this on
     * the mode is run at a higher grid and the field it computed is sent
     * whole, to be blitted and filtered — smooth curves instead of stairs.
     */
    var smooth by mutableStateOf(false)
        private set

    /**
     * How many rows of cells to fit on the screen — the app's resolution.
     *
     * This is the single number that decides how coarse everything looks. At
     * 40 rows the tablet gets a 118x34 grid, and the modes that draw at cell
     * resolution rather than into braille dots — Needle's dial and pointer,
     * VU's meters — are working with 21x42 pixel pixels. Needle's shaft is
     * about two cells wide whatever the grid, so at 40 rows it is a 42-pixel
     * bar, and it reads as a broken mode rather than a coarse one.
     *
     * More rows means smaller glyphs and a finer picture, at the cost of more
     * cells to compute and draw. Which trade is right depends on the screen
     * and on how far away it is, so it is a setting rather than a constant.
     */
    val ROW_CHOICES = listOf(24, 32, 40, 56, 72)

    var targetRows by mutableStateOf(40)
        private set

    /**
     * Cells the field path may ask a mode for, whatever the screen. A ceiling
     * against an enormous display, not the working limit — [fieldScale] is
     * what actually decides, and it decides by timing the renders.
     */
    private const val FIELD_CELL_BUDGET = 160_000

    /**
     * The field multiplier adapts, because the right one is per mode.
     *
     * Measured at 4x the tablet's grid: Chladni costs 1.4 ms a frame, Tunnel
     * In 11.5. Same cell count — the difference is that Chladni's half-blocks
     * carry two picture rows per cell where braille carries eight, so the
     * braille modes are filling four times the pixels and doing more work per
     * pixel to do it. A fixed multiplier is therefore either wasted detail on
     * one or a dropped frame on the other.
     *
     * So it is a control loop on the measured render time instead: climb while
     * frames are cheap, back off when they are not. The gap between the two
     * thresholds is hysteresis — without it the scale oscillates every frame
     * around whatever value happens to sit on the boundary.
     */
    private const val FIELD_MS_HIGH = 14.0
    private const val FIELD_MS_LOW = 6.0
    private const val FIELD_SCALE_MAX = 6
    private var fieldScale = 2
    private var renderEma = 0.0

    /** Every theme's colours, for the picker. Empty until [loadSwatches]. */
    var swatches: List<ThemeSwatch> by mutableStateOf(emptyList())
        private set

    /**
     * Fetches the picker's colours, once, on the Python thread.
     *
     * Called when the theme picker opens rather than at startup: it is 70 ms
     * of ramp interpolation that most sessions never need, and 70 ms added to
     * a launch is 70 ms of black screen.
     */
    fun loadSwatches() {
        if (swatches.isNotEmpty()) return
        val e = engine ?: return
        scope.launch {
            try {
                swatches = e.swatches()
            } catch (t: Throwable) {
                // The picker falls back to plain names; it is not worth an
                // error screen over, and the names still select correctly.
                Log.w("spektr", "could not read theme swatches", t)
            }
        }
    }

    /** Grid dimensions in cells, written by the view, read by the render loop. */
    var gridW by mutableStateOf(0)
        private set
    var gridH by mutableStateOf(0)
        private set

    private var prefs: SharedPreferences? = null
    private var renderJob: kotlinx.coroutines.Job? = null

    /** Read from the manifest flag rather than BuildConfig, which this module does not generate. */
    private var debuggable = false

    /** Builds the engine once per process. Interpreter + numpy import is ~0.5 s, so the UI shows a loading state meanwhile. */
    fun start(context: Context) {
        if (engine != null || error != null) return
        val store = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        prefs = store
        debuggable = (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
        scope.launch {
            engine = try {
                val e = PyEngine.create(context)

                // A saved name is not trusted to still exist: modes get
                // renamed and themes get dropped, and a build that starts on a
                // name the engine no longer has would fail on the first render
                // with nothing on screen to explain it.
                mode = store.getString(KEY_MODE, null)?.takeIf { it in e.modes } ?: DEFAULT_MODE
                oled = store.getBoolean(KEY_OLED, false)
                sensitivity = e.setSensitivity(store.getFloat(KEY_SENSITIVITY, 1.0f))
                smooth = store.getBoolean(KEY_SMOOTH, false)
                targetRows = store.getInt(KEY_ROWS, 40).let { r ->
                    if (r in ROW_CHOICES) r else 40
                }
                e.setFieldMode(smooth)
                val wanted = store.getString(KEY_THEME, null)?.takeIf { it in e.themes } ?: DEFAULT_THEME
                val loaded = e.useTheme(wanted, oled)
                if (loaded != null) {
                    theme = wanted
                    palette = loaded
                } else {
                    theme = DEFAULT_THEME
                    palette = e.useTheme(DEFAULT_THEME, oled)
                        ?: throw IllegalStateException("no theme named '$DEFAULT_THEME'")
                }
                e
            } catch (t: Throwable) {
                Log.e("spektr", "engine failed to start", t)
                // The class name and the first frame inside our own code, not
                // bare `t.message` — which is null for a NullPointerException,
                // so the screen read "java.lang.NullPointerException" and said
                // nothing whatever about where. With no device on adb, this
                // string is the entire diagnosis, so it has to carry its own
                // location.
                val here = t.stackTrace.firstOrNull { it.className.startsWith("dev.spektr") }
                error = buildString {
                    append(t::class.java.name)
                    t.message?.let { append(": ").append(it) }
                    here?.let {
                        append("\n  at ").append(it.methodName)
                        append(" (").append(it.fileName).append(":").append(it.lineNumber).append(")")
                    }
                    t.cause?.let {
                        append("\n  caused by ").append(it::class.java.simpleName)
                        it.message?.let { m -> append(": ").append(m) }
                    }
                }
                null
            }
        }
    }

    /** RECORD_AUDIO refused: say so on screen rather than drawing silence. */
    fun reportNoAudioPermission() {
        error = "Microphone permission refused.\n" +
            "AudioPlaybackCapture needs RECORD_AUDIO as well as the screen-capture " +
            "consent — without it there is nothing to draw."
    }

    fun setGrid(w: Int, h: Int) {
        gridW = w
        gridH = h
    }

    // `selectMode`/`selectTheme`, not `setMode`/`setTheme`: `var mode` already
    // compiles to a JVM `setMode(String)`, and a function of the same name is
    // a platform declaration clash rather than an overload.
    fun selectMode(name: String) {
        val e = engine ?: return
        if (name == mode || name !in e.modes) return
        mode = name
        // Back to a scale the next mode is certain to afford. Climbing again
        // takes a few frames; inheriting a cheap mode's 6x into an expensive
        // one costs a visible stall on the very first frame of the switch.
        fieldScale = 2
        renderEma = 0.0
        prefs?.edit()?.putString(KEY_MODE, name)?.apply()
    }

    /**
     * Switches theme on the Python thread and adopts its colours.
     *
     * [theme] is set optimistically so the chrome names the theme the moment
     * it is tapped, but [palette] only changes when Python has actually
     * switched — a tap that fails leaves the old colours and reverts the name
     * rather than showing a theme the grid is not drawn in.
     */
    fun selectTheme(name: String) {
        val e = engine ?: return
        if (name == theme || name !in e.themes) return
        val previous = theme
        theme = name
        scope.launch {
            val p = e.useTheme(name, oled)
            if (p == null) {
                Log.w("spektr", "theme '$name' was refused by the engine")
                theme = previous
            } else {
                palette = p
                prefs?.edit()?.putString(KEY_THEME, name)?.apply()
            }
        }
    }

    // `useOled`/`useSensitivity` for the same reason as `selectMode`: the
    // properties already compile to JVM setters of those names.
    fun useOled(on: Boolean) {
        val e = engine ?: return
        if (on == oled) return
        oled = on
        scope.launch {
            e.setOled(on)?.let { palette = it }
            prefs?.edit()?.putBoolean(KEY_OLED, on)?.apply()
        }
    }

    /** Clamps a wanted multiplier to something this grid can afford at all. */
    private fun capScale(want: Int, w: Int, h: Int): Int {
        if (w <= 0 || h <= 0) return 1
        var s = want.coerceIn(1, FIELD_SCALE_MAX)
        while (s > 1 && w.toLong() * h * s * s > FIELD_CELL_BUDGET) s--
        return s
    }

    /**
     * One step of the control loop described on [fieldScale].
     *
     * Driven by a running average rather than the last frame. Individual
     * renders scatter — Chladni at 4x measured 5.7 ms one frame and 10.7 the
     * next — so a per-frame rule steps up and down constantly, and every step
     * changes the field's shape and reallocates the bitmap. The average is
     * cleared after a change because the frames either side of it are
     * measurements of different things.
     */
    private fun adaptScale(renderMs: Double, w: Int, h: Int) {
        renderEma = if (renderEma <= 0.0) renderMs else renderEma * 0.9 + renderMs * 0.1
        if (renderEma > FIELD_MS_HIGH && fieldScale > 1) {
            fieldScale--
            renderEma = 0.0
        } else if (renderEma < FIELD_MS_LOW && fieldScale < FIELD_SCALE_MAX) {
            // Only climb if the next step is actually available on this grid,
            // or a capped scale reads as "still cheap" forever and the counter
            // runs away from what is being drawn.
            if (capScale(fieldScale + 1, w, h) > fieldScale) {
                fieldScale++
                renderEma = 0.0
            }
        }
    }

    fun useRows(rows: Int) {
        if (rows == targetRows || rows !in ROW_CHOICES) return
        targetRows = rows
        // The grid is about to change shape, so whatever the control loop had
        // learned about render cost at the old one is no longer about this.
        fieldScale = 2
        renderEma = 0.0
        prefs?.edit()?.putInt(KEY_ROWS, rows)?.apply()
    }

    fun useSmooth(on: Boolean) {
        val e = engine ?: return
        if (on == smooth) return
        smooth = on
        scope.launch {
            e.setFieldMode(on)
            prefs?.edit()?.putBoolean(KEY_SMOOTH, on)?.apply()
        }
    }

    fun useSensitivity(value: Float) {
        val e = engine ?: return
        scope.launch {
            val settled = e.setSensitivity(value)
            sensitivity = settled
            prefs?.edit()?.putFloat(KEY_SENSITIVITY, settled)?.apply()
        }
    }

    /** Step to the next or previous mode in the offered list, wrapping. */
    fun cycleMode(step: Int) {
        val e = engine ?: return
        val at = e.modes.indexOf(mode)
        if (at < 0) return
        selectMode(e.modes[((at + step) % e.modes.size + e.modes.size) % e.modes.size])
    }

    /** Called from the capture thread; the push hops to the Python thread. */
    fun push(pcm: ByteArray) {
        scope.launch { engine?.push(pcm) }
    }

    /**
     * Whether the app's own window is on screen.
     *
     * The home screen previews the selected mode, so frames are wanted while
     * the UI is up and not only while capturing — but a loop that ran on
     * whichever of those started first would also keep running after both had
     * finished, burning a core in the background for a picture nobody can
     * see. Both conditions are held here and the loop follows their union.
     */
    fun setUiVisible(visible: Boolean) {
        uiVisible = visible
        syncRendering()
    }

    private var uiVisible = false

    fun syncRendering() {
        if (uiVisible || CaptureController.state == CaptureController.State.Capturing) {
            startRendering()
        } else {
            stopRendering()
        }
    }

    fun startRendering() {
        if (renderJob?.isActive == true) return
        renderJob = scope.launch {
            var tick = 0
            while (isActive) {
                val began = System.nanoTime()
                val e = engine
                if (e != null && gridW > 0 && gridH > 0) {
                    try {
                        val s = if (smooth) capScale(fieldScale, gridW, gridH) else 1
                        val t0 = System.nanoTime()
                        e.render(mode, gridW * s, gridH * s)?.let { lastFrame = it }
                        if (smooth) adaptScale((System.nanoTime() - t0) / 1e6, gridW, gridH)
                    } catch (t: Throwable) {
                        Log.w("spektr", "render failed", t)
                    }
                    // How the picture is behaving is decided by numbers that
                    // never leave the Python side, and on a tablet logcat is
                    // the only way to read them. Once a second, debug builds
                    // only — a release build should not narrate itself.
                    if (debuggable && ++tick >= 30) {
                        tick = 0
                        runCatching {
                            val s = e.stats()
                            if (s.size >= 8) Log.i(
                                "spektr",
                                ("%s%s: %.1f fps  dt %.1f ms  render %.1f/%.1f ms  " +
                                    "energy %.3f  onsets %.1f/s  band %.2f  sample %.3f")
                                    .format(
                                        mode,
                                        if (smooth) " [smooth x$fieldScale]" else "",
                                        s[0], s[1], s[6], s[7], s[2], s[3], s[4], s[5],
                                    )
                            )
                        }
                    }
                }
                // Sleep the remainder of the frame, not a whole frame on top of
                // it. Waiting FRAME_MS *after* the work makes the period
                // render + 33 ms — measured on the tablet at 20 ms a render,
                // that is 53 ms and 19 fps out of a loop asking for 30.
                val spent = (System.nanoTime() - began) / 1_000_000
                delay((FRAME_MS - spent).coerceAtLeast(1L))
            }
        }
    }

    fun stopRendering() {
        renderJob?.cancel()
        renderJob = null
    }
}
