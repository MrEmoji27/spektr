package dev.spektr

import android.content.Context
import android.graphics.Color
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python

/**
 * The colours of one theme, as Android ints.
 *
 * A value class rather than fields on [PyEngine] so that switching theme is a
 * single assignment of a whole consistent set. The renderer reads the ramp and
 * the background on the same frame, and handing it a new ramp against an old
 * background — even for one frame — is a visible flash.
 */
data class Palette(val ramp: IntArray, val bg: Int, val fg: Int) {
    // IntArray gives identity equals/hashCode, which would make every
    // recomposition see a "new" palette. Compare by content.
    override fun equals(other: Any?): Boolean =
        this === other || (other is Palette && ramp.contentEquals(other.ramp) && bg == other.bg && fg == other.fg)

    override fun hashCode(): Int = 31 * (31 * ramp.contentHashCode() + bg) + fg
}

/** A theme as the picker draws it: its name, its chrome, and a few ramp colours. */
data class ThemeSwatch(val name: String, val bg: Int, val fg: Int, val colors: IntArray) {
    override fun equals(other: Any?): Boolean =
        this === other || (other is ThemeSwatch && name == other.name && colors.contentEquals(other.colors))

    override fun hashCode(): Int = 31 * name.hashCode() + colors.contentHashCode()
}

/**
 * The Kotlin side of the one boundary that matters in this port: a Chaquopy
 * `spektr_android.Engine` plus everything Kotlin reads out of Python.
 *
 * Every call here must run on the single Python thread owned by
 * [EngineManager]; Chaquopy's API is not thread-safe and push, render and a
 * theme switch must serialise against each other.
 *
 * The ramp colours come from Python, not from a hardcoded table: `Palette.hexes`
 * is what the desktop's renderers consume, so the APK draws the same colours
 * by construction.
 */
class PyEngine private constructor(
    private val engine: PyObject,
    /** Every mode the picker may offer. Hidden octant variants are not among them. */
    val modes: List<String>,
    val themes: List<String>,
) {
    /** Interleaved float32 little-endian stereo PCM — `AudioRecord`'s ENCODING_PCM_FLOAT. */
    fun push(pcm: ByteArray) {
        engine.callAttr("push", pcm)
    }

    /**
     * One frame of [mode] at this grid size.
     *
     * The mode is a per-call argument rather than engine state because that is
     * how the Python side already works — and because it means switching mode
     * is not a state change that can fail or race a render. Python drops the
     * mode's scratch when the name changes, so no mode ever inherits another's
     * arrays.
     */
    fun render(mode: String, w: Int, h: Int): FrameBuf? {
        val buf = engine.callAttr("render", mode, w, h)
        return FrameBuf.parse(buf.toJava(ByteArray::class.java))
    }

    /**
     * Every theme's colours, for painting the picker. Costs about 70 ms —
     * every ramp gets interpolated — so it is fetched once, off a frame.
     */
    fun swatches(): List<ThemeSwatch> =
        engine.callAttr("theme_swatches").asList().map { row ->
            val cells = row.asList().map { it.toString() }
            ThemeSwatch(
                name = cells[0],
                bg = hexToArgb(cells[1]),
                fg = hexToArgb(cells[2]),
                colors = IntArray(cells.size - 3) { hexToArgb(cells[it + 3]) },
            )
        }

    /** Switches theme and returns its colours, or null if there is no such theme. */
    fun useTheme(name: String): Palette? {
        val hexes = engine.callAttr("use_theme", name)?.asList() ?: return null
        if (hexes.size < 3) {
            Log.w(TAG, "theme '$name' gave ${hexes.size} colours, expected bg + fg + ramp")
            return null
        }
        return Palette(
            ramp = IntArray(hexes.size - 2) { hexToArgb(hexes[it + 2].toString()) },
            bg = hexToArgb(hexes[0].toString()),
            fg = hexToArgb(hexes[1].toString()),
        )
    }

    companion object {
        private const val TAG = "spektr"

        fun create(context: Context): PyEngine {
            val py = Python.getInstance()
            val engine = py.getModule("spektr_android").callAttr("Engine")

            // Lists, and always via callAttr.
            //
            // Chaquopy's `PyObject.get` is *attribute* access, so the previous
            // version's `BUILTIN.get("gruvbox")` asked a dict for an attribute
            // named gruvbox, got null, and the `!!` after it threw a
            // NullPointerException before the first frame. `callAttr` is
            // unambiguous — always a method call — so the whole class of
            // mistake is gone rather than fixed once.
            val modes = engine.callAttr("mode_names").asList().map { it.toString() }
            val themes = engine.callAttr("theme_names").asList().map { it.toString() }
            require(modes.isNotEmpty()) { "the engine offers no modes" }
            require(themes.isNotEmpty()) { "the engine offers no themes" }

            Log.i(TAG, "engine up: ${modes.size} modes, ${themes.size} themes")
            return PyEngine(engine, modes, themes)
        }

        fun hexToArgb(hex: String): Int {
            val s = hex.trim().removePrefix("#")
            val r = s.substring(0, 2).toInt(16)
            val g = s.substring(2, 4).toInt(16)
            val b = s.substring(4, 6).toInt(16)
            return Color.rgb(r, g, b)
        }
    }
}
