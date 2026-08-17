package dev.spektr

import android.content.Context
import android.graphics.Color
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python

/**
 * The Kotlin side of the one boundary that matters in this port: a Chaquopy
 * `spektr_android.Engine` plus everything Kotlin reads out of Python.
 *
 * Every call here must run on the single Python thread owned by
 * [EngineManager]; Chaquopy's API is not thread-safe and push and render must
 * serialise against each other.
 *
 * The ramp colours come from Python, not from a hardcoded table: `Palette.hexes`
 * is what the desktop's renderers consume, so the APK draws the same colours
 * by construction. Reading them across the boundary once, at construction, is
 * the one palette crossing per session.
 */
class PyEngine private constructor(
    private val engine: PyObject,
    val mode: String,
    val theme: String,
    val ramp: IntArray,
    val bgColor: Int,
    val fgColor: Int,
) {
    /** Interleaved float32 little-endian stereo PCM — `AudioRecord`'s ENCODING_PCM_FLOAT. */
    fun push(pcm: ByteArray) {
        engine.callAttr("push", pcm)
    }

    fun render(w: Int, h: Int): FrameBuf? {
        val buf = engine.callAttr("render", mode, w, h)
        return FrameBuf.parse(buf.toJava(ByteArray::class.java))
    }

    companion object {
        private const val TAG = "spektr"

        fun create(context: Context): PyEngine {
            val py = Python.getInstance()
            val engine = py.getModule("spektr_android").callAttr("Engine")

            val themeName = EngineManager.THEME
            if (!engine.callAttr("set_theme", themeName).toBoolean()) {
                throw IllegalStateException("no theme named '$themeName'")
            }

            // Colours arrive as flat lists of "#rrggbb" from method calls.
            //
            // The previous version reached into Python's own objects from here
            // — `BUILTIN.get(name)`, `palette.get("hexes")` — and that is what
            // put the NullPointerException on the tablet. Chaquopy's
            // `PyObject.get` is *attribute* access, so asking a dict for
            // `.get("gruvbox")` returns null, and the `!!` after it threw
            // before a single frame was drawn.
            //
            // `callAttr` is unambiguous — always a method call — so this whole
            // class of mistake is gone rather than fixed once.
            val rampHex = engine.callAttr("ramp_hexes").asList()
            require(rampHex.isNotEmpty()) { "the palette handed back an empty ramp" }
            val ramp = IntArray(rampHex.size) { hexToArgb(rampHex[it].toString()) }

            val chrome = engine.callAttr("chrome_hexes").asList()
            require(chrome.size == 2) { "chrome_hexes gave ${chrome.size} colours, expected 2" }
            val bg = hexToArgb(chrome[0].toString())
            val fg = hexToArgb(chrome[1].toString())

            Log.i(TAG, "engine up: $themeName, ${ramp.size} ramp steps, bg $bg")
            return PyEngine(engine, EngineManager.MODE, themeName, ramp, bg, fg)
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
