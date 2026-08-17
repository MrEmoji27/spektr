package dev.spektr

import android.content.Context
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext

/**
 * App-scoped owner of the Python engine and the frame pipeline.
 *
 * One thread ("spektr-py") carries every Python call: construction, pushes
 * from the capture thread and one render per frame tick. That serialises the
 * engine's only shared mutable state (its ring buffer and mode scratch)
 * without locks, and it keeps the JNI crossing off the main thread.
 *
 * The render loop is paced at ~30 fps — the design's target rate, half the
 * desktop's, because every frame costs a JNI crossing and the panel is
 * watched from across a room. Frames are produced even while the user has not
 * consented yet, so the first post-consent picture is not a blank grid.
 */
object EngineManager {

    // v1: one hardcoded mode and one hardcoded theme, per the design's build
    // order. Kaleidoscope is a half-block mode: it carries the background
    // plane, so the renderer's plane handling is exercised on day one.
    const val MODE = "Kaleidoscope"
    const val THEME = "gruvbox"
    private const val FRAME_MS = 33L

    private val pyContext = newSingleThreadContext("spektr-py")
    private val scope = CoroutineScope(pyContext + SupervisorJob())

    var engine: PyEngine? by mutableStateOf(null)
        private set
    var error: String? by mutableStateOf(null)
        private set

    /** The newest parsed frame, drawn by [GridView]. */
    var lastFrame: FrameBuf? by mutableStateOf(null)
        private set

    /** Grid dimensions in cells, written by the view, read by the render loop. */
    var gridW by mutableStateOf(0)
        private set
    var gridH by mutableStateOf(0)
        private set

    private var renderJob: kotlinx.coroutines.Job? = null

    /** Builds the engine once per process. Interpreter + numpy import is seconds, so the UI shows a loading state meanwhile. */
    fun start(context: Context) {
        if (engine != null || error != null) return
        scope.launch {
            engine = try {
                PyEngine.create(context)
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

    /** Called from the capture thread; the push hops to the Python thread. */
    fun push(pcm: ByteArray) {
        scope.launch { engine?.push(pcm) }
    }

    fun startRendering() {
        if (renderJob?.isActive == true) return
        renderJob = scope.launch {
            while (isActive) {
                val began = System.nanoTime()
                val e = engine
                if (e != null && gridW > 0 && gridH > 0) {
                    try {
                        e.render(gridW, gridH)?.let { lastFrame = it }
                    } catch (t: Throwable) {
                        Log.w("spektr", "render failed", t)
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
