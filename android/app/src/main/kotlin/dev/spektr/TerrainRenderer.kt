package dev.spektr

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.cos
import kotlin.math.sin

/**
 * The terrain view: the mode's own float field drawn as a lit, displaced
 * surface instead of a flat blit.
 *
 * The pipeline is the one `docs/internal/android-3d.md` argued for. Python
 * ships the frame as float heights (wire v2, planes == 4); this class uploads
 * them as an R32F texture once per frame and draws one static grid mesh whose
 * vertex shader displaces Y by the sampled height. Colour still comes from
 * spektr's 64-entry ramp — uploaded as a 1D texture, so all themes work
 * unchanged — and lighting is computed in the fragment shader from height
 * taps either side of the fragment, which is why the field crosses the wire
 * as floats: 64-step heights would terrace every slope the light touches.
 *
 * Everything here runs on the GL thread; frames arrive from the render loop's
 * thread through [submit] and are picked up at the next draw. One texture
 * upload and one draw call per frame replaces what FieldBlitter spent on CPU
 * walking pixels.
 */
class TerrainRenderer : GLSurfaceView.Renderer {

    @Volatile private var pendingFrame: FrameBuf? = null
    @Volatile private var pendingRamp: IntArray? = null
    @Volatile private var pendingBg: Int = 0xFF000000.toInt()

    private var program = 0
    private var heightTex = 0
    private var rampTex = 0
    private var verts: FloatBuffer? = null
    private var indices: ByteBuffer? = null
    private var indexCount = 0

    private var texW = 0
    private var texH = 0
    private var heightData: FloatBuffer? = null

    /** Session clock for the camera's drift, started when the renderer is. */
    private val t0 = System.nanoTime()

    private val projection = FloatArray(16)
    private val view = FloatArray(16)
    private val vp = FloatArray(16)

    // Uniform locations, resolved once at link time.
    private var uVP = 0
    private var uAmp = 0
    private var uTexel = 0
    private var uLight = 0
    private var uBg = 0

    /** Latest frame from the render loop; swapped atomically, consumed on GL. */
    fun submit(frame: FrameBuf) {
        if (frame.isFloatField) {
            pendingFrame = frame
        }
    }

    fun submitPalette(palette: Palette) {
        pendingRamp = palette.ramp
        pendingBg = palette.bg
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        program = buildProgram()
        uVP = GLES30.glGetUniformLocation(program, "uVP")
        uAmp = GLES30.glGetUniformLocation(program, "uAmp")
        uTexel = GLES30.glGetUniformLocation(program, "uTexel")
        uLight = GLES30.glGetUniformLocation(program, "uLight")
        uBg = GLES30.glGetUniformLocation(program, "uBg")

        heightTex = makeTexture()
        rampTex = makeTexture()
        buildMesh()
        uploadRamp()
        // A first frame may already be waiting; without this the surface
        // draws an empty world until the next render tick.
        pendingFrame?.let { uploadHeights(it) }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
        aspect(width.toFloat() / height.toFloat().coerceAtLeast(1f))
    }

    override fun onDrawFrame(gl: GL10?) {
        pendingFrame?.let { if (it.w != texW || it.h != texH) resetTextures(it.w, it.h) }
        pendingFrame?.let { uploadHeights(it) }
        pendingRamp?.let { uploadRamp(it); pendingRamp = null }
        val bg = pendingBg

        val t = (System.nanoTime() - t0) / 1_000_000_000f
        // Turntable around the origin: eye on a sphere whose elevation and
        // yaw come from the same angles the rotations use, so the origin
        // lands dead-centre. An earlier cut offset the eye sideways, and the
        // landscape slid into a corner of the screen.
        val pitchDeg = 36f
        val radius = 2.45f
        val elev = Math.toRadians(pitchDeg.toDouble()).toFloat()
        val az = 0.9f + sin(t * 0.05f) * 0.25f            // slow drift, never still
        val horiz = cos(elev) * radius
        val ex = sin(az) * horiz
        val ey = (sin(elev) * radius).toFloat()
        val ez = cos(az) * horiz
        Matrix.setIdentityM(view, 0)
        Matrix.rotateM(view, 0, pitchDeg, 1f, 0f, 0f)
        Matrix.rotateM(view, 0, az * 57.2958f - 90f, 0f, 1f, 0f)
        Matrix.translateM(view, 0, -ex, -ey, -ez)
        Matrix.multiplyMM(vp, 0, projection, 0, view, 0)

        GLES30.glClearColor(
            ((bg shr 16) and 0xFF) / 255f,
            ((bg shr 8) and 0xFF) / 255f,
            (bg and 0xFF) / 255f,
            1f,
        )
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        if (indexCount == 0 || texW == 0) return
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(uVP, 1, false, vp, 0)
        GLES30.glUniform1f(uAmp, AMP)
        GLES30.glUniform2f(uTexel, 1f / texW, 1f / texH)
        GLES30.glUniform3f(uLight, LIGHT_X, LIGHT_Y, LIGHT_Z)
        GLES30.glUniform3f(
            uBg,
            ((bg shr 16) and 0xFF) / 255f,
            ((bg shr 8) and 0xFF) / 255f,
            (bg and 0xFF) / 255f,
        )

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GL_TEXTURE_BIND_TARGET, heightTex)
        GLES30.glUniform1i(uHeightLoc(), 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GL_TEXTURE_BIND_TARGET, rampTex)
        GLES30.glUniform1i(uRampLoc(), 1)

        val stride = 2 * FLOAT_SIZE
        verts?.position(0)
        GLES30.glVertexAttribPointer(ATTR_UV, 2, GLES30.GL_FLOAT, false, stride, verts)
        GLES30.glEnableVertexAttribArray(ATTR_UV)
        GLES30.glDrawElements(
            GLES30.GL_TRIANGLES, indexCount, GLES30.GL_UNSIGNED_SHORT, indices
        )
        GLES30.glDisableVertexAttribArray(ATTR_UV)
    }

    // ── frame & palette uploads ────────────────────────────────────────────

    private fun resetTextures(w: Int, h: Int) {
        texW = w
        texH = h
        heightData = ByteBuffer
            .allocateDirect(w * h * FLOAT_SIZE)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        GLES30.glBindTexture(GL_TEXTURE_BIND_TARGET, heightTex)
        GLES30.glTexImage2D(
            GL_TEXTURE_BIND_TARGET, 0, GLES30.GL_R32F, w, h, 0,
            GLES30.GL_RED, GLES30.GL_FLOAT, null,
        )
    }

    private fun uploadHeights(frame: FrameBuf) {
        val floats = frame.fvals ?: return
        if (frame.w != texW || frame.h != texH) resetTextures(frame.w, frame.h)
        val buf = heightData ?: return
        buf.clear()
        buf.put(floats)
        buf.position(0)
        GLES30.glBindTexture(GL_TEXTURE_BIND_TARGET, heightTex)
        GLES30.glTexSubImage2D(
            GL_TEXTURE_BIND_TARGET, 0, 0, 0, texW, texH,
            GLES30.GL_RED, GLES30.GL_FLOAT, buf,
        )
        pendingFrame = null
    }

    private fun uploadRamp(ramp: IntArray? = null) {
        val colors = ramp ?: pendingRamp ?: return
        val rgba = ByteBuffer.allocateDirect(colors.size * 4)
        for (c in colors) {
            rgba.put(((c shr 16) and 0xFF).toByte())
            rgba.put(((c shr 8) and 0xFF).toByte())
            rgba.put((c and 0xFF).toByte())
            rgba.put(0xFF.toByte())
        }
        rgba.position(0)
        GLES30.glBindTexture(GL_TEXTURE_BIND_TARGET, rampTex)
        GLES30.glTexImage2D(
            GL_TEXTURE_BIND_TARGET, 0, GLES30.GL_RGBA, colors.size, 1, 0,
            GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, rgba,
        )
        pendingRamp = null
    }

    // ── geometry ───────────────────────────────────────────────────────────
    //
    // A static grid over uv [0..1]^2. Resolution chosen so a short index
    // still works: 193 x 109 vertices stays under the unsigned-short limit.

    private fun buildMesh() {
        val gx = GRID_X
        val gz = GRID_Z
        val vertCount = gx * gz
        val v = ByteBuffer
            .allocateDirect(vertCount * 2 * FLOAT_SIZE)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        for (z in 0 until gz) {
            for (x in 0 until gx) {
                v.put(x.toFloat() / (gx - 1))
                v.put(z.toFloat() / (gz - 1))
            }
        }
        v.position(0)
        verts = v

        val quads = (gx - 1) * (gz - 1)
        indexCount = quads * 6
        val idx = ByteBuffer
            .allocateDirect(indexCount * SHORT_SIZE)
            .order(ByteOrder.nativeOrder())
        for (z in 0 until gz - 1) {
            for (x in 0 until gx - 1) {
                val a = (z * gx + x).toShort()
                val b = (z * gx + x + 1).toShort()
                val c = ((z + 1) * gx + x).toShort()
                val d = ((z + 1) * gx + x + 1).toShort()
                idx.putShort(a).putShort(c).putShort(b)
                idx.putShort(b).putShort(c).putShort(d)
            }
        }
        idx.position(0)
        indices = idx
    }

    // ── GL plumbing ────────────────────────────────────────────────────────

    private fun aspect(ratio: Float) {
        val near = 0.1f
        val far = 12f
        Matrix.perspectiveM(projection, 0, 42f, ratio, near, far)
    }

    private fun makeTexture(): Int {
        val tex = IntArray(1)
        GLES30.glGenTextures(1, tex, 0)
        GLES30.glBindTexture(GL_TEXTURE_BIND_TARGET, tex[0])
        GLES30.glTexParameteri(GL_TEXTURE_BIND_TARGET, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GL_TEXTURE_BIND_TARGET, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GL_TEXTURE_BIND_TARGET, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GL_TEXTURE_BIND_TARGET, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        return tex[0]
    }

    private fun uHeightLoc() = GLES30.glGetUniformLocation(program, "uHeight")
    private fun uRampLoc() = GLES30.glGetUniformLocation(program, "uRamp")

    private fun buildProgram(): Int {
        val vs = """
            #version 300 es
            precision highp float;
            layout(location = $ATTR_UV) in vec2 aUV;
            uniform mat4 uVP;
            uniform sampler2D uHeight;
            uniform float uAmp;
            out vec2 vUV;
            out vec3 vWorld;
            void main() {
                vUV = aUV;
                float hh = texture(uHeight, aUV).r;
                vec3 p = vec3((aUV.x - 0.5) * 2.0, hh * uAmp, (aUV.y - 0.5) * 1.15);
                vWorld = p;
                gl_Position = uVP * vec4(p, 1.0);
            }
        """.trimIndent()

        val fs = """
            #version 300 es
            precision highp float;
            in vec2 vUV;
            in vec3 vWorld;
            out vec4 fragColor;
            uniform sampler2D uHeight;
            uniform sampler2D uRamp;
            uniform vec2 uTexel;
            uniform vec3 uLight;
            uniform vec3 uBg;
            uniform float uAmp;
            vec3 cameraDir() {
                return normalize(vec3(0.0, 1.2, -0.8) - vWorld);
            }
            void main() {
                float hl = texture(uHeight, vUV - vec2(uTexel.x, 0.0)).r;
                float hr = texture(uHeight, vUV + vec2(uTexel.x, 0.0)).r;
                float hd = texture(uHeight, vUV - vec2(0.0, uTexel.y)).r;
                float hu = texture(uHeight, vUV + vec2(0.0, uTexel.y)).r;
                vec3 n = normalize(vec3((hl - hr) * uAmp * SLOPE, 1.0, (hd - hu) * uAmp * SLOPE));
                vec3 l = normalize(uLight);
                float diff = max(dot(n, l), 0.0);
                float spec = pow(max(dot(reflect(-l, n), cameraDir()), 0.0), 24.0) * 0.15;
                float shade = AMBIENT + DIFFUSE * diff;
                vec3 base = texture(uRamp, vec2(texture(uHeight, vUV).r, 0.5)).rgb;
                vec3 col = base * shade + vec3(spec);
                float edge = smoothstep(0.0, 0.06, texture(uHeight, vUV).r);
                fragColor = vec4(mix(uBg, col, edge), 1.0);
            }
        """.trimIndent()
            .replace("SLOPE", SLOPE_STR)
            .replace("AMBIENT", AMBIENT_STR)
            .replace("DIFFUSE", DIFFUSE_STR)

        val vsh = compile(GLES30.GL_VERTEX_SHADER, vs)
        val fsh = compile(GLES30.GL_FRAGMENT_SHADER, fs)
        val prog = GLES30.glCreateProgram()
        GLES30.glAttachShader(prog, vsh)
        GLES30.glAttachShader(prog, fsh)
        GLES30.glLinkProgram(prog)
        val linked = IntArray(1)
        GLES30.glGetProgramiv(prog, GLES30.GL_LINK_STATUS, linked, 0)
        if (linked[0] == 0) {
            val log = GLES30.glGetProgramInfoLog(prog)
            GLES30.glDeleteProgram(prog)
            error("terrain shader link failed: $log")
        }
        GLES30.glDeleteShader(vsh)
        GLES30.glDeleteShader(fsh)
        return prog
    }

    private fun compile(type: Int, source: String): Int {
        val sh = GLES30.glCreateShader(type)
        GLES30.glShaderSource(sh, source)
        GLES30.glCompileShader(sh)
        val ok = IntArray(1)
        GLES30.glGetShaderiv(sh, GLES30.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) {
            val log = GLES30.glGetShaderInfoLog(sh)
            GLES30.glDeleteShader(sh)
            error("terrain shader compile failed: $log")
        }
        return sh
    }

    companion object {
        private const val ATTR_UV = 0
        private const val GL_TEXTURE_BIND_TARGET = GLES30.GL_TEXTURE_2D
        private const val FLOAT_SIZE = 4
        private const val SHORT_SIZE = 2
        private const val GRID_X = 193
        private const val GRID_Z = 109

        /** World-units of displacement at height 1.0. */
        private const val AMP = 0.45f

        /** How strongly slopes respond to the light, relative to flat. */
        private const val SLOPE_STR = "2.4"
        private const val AMBIENT_STR = "0.55"
        private const val DIFFUSE_STR = "0.60"
        private const val LIGHT_X = 0.45f
        private const val LIGHT_Y = 0.75f
        private const val LIGHT_Z = 0.50f
    }
}
