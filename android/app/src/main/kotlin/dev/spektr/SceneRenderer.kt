package dev.spektr

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * The scene view: four raymarched worlds, drawn entirely in a fragment shader.
 *
 * This replaces a height-mapped terrain renderer, and the replacement is a
 * change of kind rather than of degree. That one asked Python for a picture —
 * a float per pixel, every frame, 33 ms of numpy on the tablet — and then
 * displaced a grid mesh by it. What it could ever be was therefore one thing:
 * a lit sheet. Seen on a device it read as a strip of landscape floating in a
 * black frame, which is exactly what it was.
 *
 * Here Python ships no picture at all. A scene frame is forty floats — energy,
 * bass, mid, treble, an onset envelope, the beat phase, twenty-four bands —
 * and the shader builds the world from them. Three things follow:
 *
 * * **it fills the screen by construction.** There is no mesh to frame and no
 *   camera distance to solve: every pixel is in the scene, and the ray through
 *   it hits something or hits the sky, which is itself drawn in the theme's
 *   colours. The old renderer's entire framing problem is gone rather than
 *   fixed;
 * * **it costs almost nothing on the Python side.** 172 bytes and no arrays,
 *   against a mode's worth of numpy per frame;
 * * **shape is now free.** Fusing metaballs, an infinite tunnel, a fractured
 *   solid and a lattice running to the horizon are four `map` functions, not
 *   four geometry pipelines.
 *
 * Colour still comes from spektr's 64-entry ramp, uploaded as a texture, so
 * all fifty-odd themes work here unchanged and untested-per-theme.
 *
 * The surface is rendered at reduced resolution and scaled up by the display
 * hardware ([FIXED_LONG_SIDE]). A raymarcher is fill-bound and this panel is
 * 1536x2560; at native resolution the cost is four megapixels of sphere
 * tracing, which no phone GPU does at thirty frames. Half is invisible on an
 * organic scene and four times cheaper.
 */
class SceneRenderer : GLSurfaceView.Renderer {

    /** Latest parameters from the render loop. Read on the GL thread. */
    @Volatile private var params: FloatArray? = null

    @Volatile private var ramp: IntArray? = null
    @Volatile private var rampDirty = true
    @Volatile private var bg: Int = 0xFF000000.toInt()

    private var program = 0
    private var rampTex = 0

    private var uRes = 0
    private var uTime = 0
    private var uParams = 0
    private var uRamp = 0
    private var uBg = 0

    private val t0 = System.nanoTime()
    private var resW = 1f
    private var resH = 1f

    fun submit(frame: FrameBuf) {
        if (frame.isScene) params = frame.fvals
    }

    fun submitPalette(palette: Palette) {
        if (!palette.ramp.contentEquals(ramp)) {
            ramp = palette.ramp
            rampDirty = true
        }
        bg = palette.bg
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glDisable(GLES30.GL_CULL_FACE)

        // The context does not survive leaving the scene view and coming back,
        // so everything named here is re-made and the ramp is re-uploaded.
        // Holding the colours rather than consuming them is what makes that
        // possible; a drained queue handed the rebuilt context a black ramp.
        rampDirty = true
        program = buildProgram()
        uRes = GLES30.glGetUniformLocation(program, "uRes")
        uTime = GLES30.glGetUniformLocation(program, "uTime")
        uParams = GLES30.glGetUniformLocation(program, "uP")
        uRamp = GLES30.glGetUniformLocation(program, "uRamp")
        uBg = GLES30.glGetUniformLocation(program, "uBg")
        rampTex = makeRampTexture()
        uploadRamp()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
        resW = width.toFloat()
        resH = height.toFloat()
    }

    override fun onDrawFrame(gl: GL10?) {
        if (rampDirty) uploadRamp()
        val p = params ?: return

        GLES30.glClearColor(
            ((bg shr 16) and 0xFF) / 255f,
            ((bg shr 8) and 0xFF) / 255f,
            (bg and 0xFF) / 255f,
            1f,
        )
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glUseProgram(program)
        GLES30.glUniform2f(uRes, resW, resH)
        // The clock is the GL thread's, not Python's. Audio arrives at thirty
        // frames a second and the surface draws at sixty; taking time from the
        // frame would quantise every rotation and orbit to the slower of the
        // two and show as judder on motion that has nothing to do with sound.
        GLES30.glUniform1f(uTime, (System.nanoTime() - t0) / 1_000_000_000f)
        GLES30.glUniform1fv(uParams, FrameBuf.SCENE_FLOATS, p, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, rampTex)
        GLES30.glUniform1i(uRamp, 0)
        GLES30.glUniform3f(
            uBg,
            ((bg shr 16) and 0xFF) / 255f,
            ((bg shr 8) and 0xFF) / 255f,
            (bg and 0xFF) / 255f,
        )

        // No vertex buffer and no attributes: three vertices synthesised from
        // gl_VertexID cover the screen with one triangle. A quad would need
        // two, and the diagonal seam between them costs a second shading pass
        // on every fragment the two triangles share.
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
    }

    private fun uploadRamp() {
        val colors = ramp ?: return
        val rgba = ByteBuffer.allocateDirect(colors.size * 4).order(ByteOrder.nativeOrder())
        for (c in colors) {
            rgba.put(((c shr 16) and 0xFF).toByte())
            rgba.put(((c shr 8) and 0xFF).toByte())
            rgba.put((c and 0xFF).toByte())
            rgba.put(0xFF.toByte())
        }
        rgba.position(0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, rampTex)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, colors.size, 1, 0,
            GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, rgba,
        )
        rampDirty = false
    }

    private fun makeRampTexture(): Int {
        val tex = IntArray(1)
        GLES30.glGenTextures(1, tex, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        return tex[0]
    }

    private fun buildProgram(): Int {
        val vs = """
            #version 300 es
            precision highp float;
            void main() {
                // One oversized triangle: (-1,-1), (3,-1), (-1,3). Its
                // intersection with the viewport is exactly the screen.
                vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
                gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
            }
        """.trimIndent()

        val fs = SCENE_SHADER
            .replace("N_FLOATS", FrameBuf.SCENE_FLOATS.toString())
            .replace("N_BANDS", FrameBuf.SCENE_BANDS.toString())
            .replace("HEAD", FrameBuf.SCENE_HEAD.toString())
            .replace("MAX_STEPS", MAX_STEPS.toString())

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
            error("scene shader link failed: $log")
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
            error("scene shader compile failed: $log")
        }
        return sh
    }

    companion object {
        /**
         * Rays per pixel are the whole cost, so the surface is rendered at
         * this long side and the display scales it up. On a 1536x2560 panel
         * that is four megapixels of sphere tracing against one.
         */
        const val FIXED_LONG_SIDE = 1280

        private const val MAX_STEPS = 72

        /**
         * The whole scene family, as one fragment shader.
         *
         * One program rather than four, switched on a uniform. The branch is
         * uniform-coherent — every fragment in the frame takes the same arm —
         * so a tiled GPU costs nothing for it, and against four programs it
         * saves three compiles at surface creation and all the state changes.
         *
         * Screen coordinates are normalised by the **short** side. That single
         * choice is what makes every scene fit every device: the subject is
         * sized against the dimension that is scarce, so rotating the tablet
         * or running on a phone shows more of the world rather than cropping
         * into it.
         */
        private val SCENE_SHADER = """
            #version 300 es
            precision highp float;

            out vec4 fragColor;
            uniform vec2 uRes;
            uniform float uTime;
            uniform float uP[N_FLOATS];
            uniform sampler2D uRamp;
            uniform vec3 uBg;

            // Named reads of the block Python packs. See spektr_android.SCENES.
            #define SCENE   int(uP[0])
            #define ENERGY  uP[2]
            #define BASS    uP[3]
            #define MID     uP[4]
            #define TREBLE  uP[5]
            #define PULSE   uP[6]
            #define HARDEST uP[7]
            #define BEAT    uP[8]
            #define TEMPO   uP[9]
            #define PEAK    uP[11]
            float band(int i) { return uP[HEAD + (i % N_BANDS)]; }

            vec3 ramp(float x) { return texture(uRamp, vec2(clamp(x, 0.02, 0.98), 0.5)).rgb; }

            mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

            float sdSphere(vec3 p, float r) { return length(p) - r; }

            float sdBox(vec3 p, vec3 b, float r) {
                vec3 q = abs(p) - b;
                return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
            }

            float hash13(vec3 p) {
                return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
            }

            // ── the four worlds ──────────────────────────────────────────────
            // Each returns (distance, material) where material is a 0..1
            // coordinate along the theme's ramp. Keeping colour as one scalar
            // is what lets fifty-five themes work without a line of per-theme
            // code anywhere.

            // Seven spheres on lissajous orbits, fused by a polynomial
            // smooth-minimum. Radius and orbit both come from the spectrum, so
            // quiet music is a tight cluster and loud music flies apart; a hit
            // inflates every one of them at once.
            vec2 mapBall(vec3 p) {
                float d = 1e9;
                float m = 0.0;
                for (int i = 0; i < 7; i++) {
                    float fi = float(i);
                    float b = band(i * 3 + 1);
                    float rr = 0.17 + b * 0.26 + PULSE * 0.05;
                    float orb = 0.50 + b * 0.30 + PULSE * 0.30;
                    float sp = 0.32 + fi * 0.10;
                    vec3 c = vec3(
                        sin(uTime * sp + fi * 2.4) * orb,
                        cos(uTime * sp * 0.83 + fi * 1.7) * orb * 0.75,
                        sin(uTime * sp * 0.61 + fi * 3.1) * orb);
                    float dd = sdSphere(p - c, rr);
                    float k = 0.30 + BASS * 0.14;
                    float hh = clamp(0.5 + 0.5 * (d - dd) / k, 0.0, 1.0);
                    d = mix(d, dd, hh) - k * hh * (1.0 - hh);
                    m = mix(m, 0.25 + fi / 9.0 + b * 0.3, hh);
                }
                // A floor, so the frame is a room rather than an object in a
                // void. It ripples with the spectrum and catches the light.
                float floorY = p.y + 1.15
                    + 0.04 * sin(p.x * 3.0 + uTime) * sin(p.z * 3.0 - uTime * 0.7) * (0.3 + MID);
                if (floorY < d) return vec2(floorY, 0.07);
                return vec2(d, m);
            }

            // An endless tube seen from inside: distance is radius-minus-reach,
            // so the sign flips and the camera is always contained. The axis
            // wanders, the wall is ribbed along its length and fluted around
            // it, and the flight speed follows the tempo.
            vec2 mapHole(vec3 p) {
                float z = p.z;
                vec2 axis = vec2(
                    sin(z * 0.31 + uTime * 0.6) * 0.55 + sin(z * 0.17 - uTime * 0.35) * 0.30,
                    cos(z * 0.26 + uTime * 0.5) * 0.50);
                vec2 q = p.xy - axis;
                float r = length(q);
                float a = atan(q.y, q.x);
                float ribs = 0.09 * sin(z * 2.6 - uTime * 3.0 - BEAT * 6.2831);
                float flute = (0.05 + TREBLE * 0.13) * sin(a * 9.0 + z * 0.8 + uTime * 1.4);
                float R = 1.25 + ribs + flute - BASS * 0.28 - PULSE * 0.10;
                float m = 0.20 + 0.55 * fract(z * 0.06 - uTime * 0.05) + band(int(abs(a) * 4.0)) * 0.25;
                return vec2(R - r, m);
            }

            // One solid, tumbling: a box and a sphere crossfading into each
            // other, its surface raised into ridges by six bands and split
            // open by a high-frequency lattice on every hit.
            vec2 mapMono(vec3 p) {
                vec3 q = p;
                q.xz = rot(uTime * 0.33) * q.xz;
                q.xy = rot(uTime * 0.19) * q.xy;
                float blend = 0.35 + 0.35 * sin(uTime * 0.27);
                float d = mix(sdBox(q, vec3(0.58), 0.10), sdSphere(q, 0.84), blend);
                float ridge = 0.0;
                for (int i = 0; i < 6; i++) {
                    float f = 5.0 + float(i) * 4.5;
                    ridge += band(i * 4) * 0.030 * sin((q.x + q.y * 1.3 + q.z * 0.7) * f
                                                       + uTime * (0.8 + float(i) * 0.35));
                }
                d -= ridge;
                d += PULSE * HARDEST * 0.10 * sin(q.x * 19.0) * sin(q.y * 19.0) * sin(q.z * 19.0);
                float m = 0.30 + 0.45 * clamp(length(q) - 0.4, 0.0, 1.0) + PEAK * 0.2;
                float floorY = p.y + 1.30;
                if (floorY < d) return vec2(floorY, 0.06);
                return vec2(d, m);
            }

            // Space folded into cells, one rounded block in each, sized by a
            // band the cell picks for itself from its own coordinates. It
            // reaches the horizon in every direction, so there is no framing
            // question left to get wrong.
            vec2 mapLat(vec3 p) {
                vec3 q = p;
                q.xz = rot(uTime * 0.10) * q.xz;
                q.y += sin(uTime * 0.3) * 0.2;
                float cell = 1.5;
                vec3 id = floor(q / cell);
                vec3 c = mod(q, cell) - cell * 0.5;
                float hs = hash13(id);
                float b = band(int(hs * float(N_BANDS)));
                float s = 0.14 + b * 0.32 + PULSE * HARDEST * 0.10;
                float d = sdBox(c, vec3(s), 0.05);
                return vec2(d, 0.18 + hs * 0.35 + b * 0.42);
            }

            vec2 map(vec3 p) {
                if (SCENE == 0) return mapBall(p);
                if (SCENE == 1) return mapHole(p);
                if (SCENE == 2) return mapMono(p);
                return mapLat(p);
            }

            vec3 normalAt(vec3 p) {
                // Tetrahedron taps: four samples for a gradient instead of the
                // six a central difference needs, which on a marcher this deep
                // is a third of the whole shader's cost.
                vec2 e = vec2(1.0, -1.0) * 0.0018;
                return normalize(
                    e.xyy * map(p + e.xyy).x + e.yyx * map(p + e.yyx).x +
                    e.yxy * map(p + e.yxy).x + e.xxx * map(p + e.xxx).x);
            }

            // The sky is drawn in the theme, not left black. An object in a
            // black frame reads as a floating fragment; the same object under
            // a graded sky with a glow behind it reads as a place.
            vec3 sky(vec2 uv, vec3 rd) {
                float up = 0.5 + 0.5 * rd.y;
                vec3 c = mix(mix(uBg, ramp(0.10), 0.55), uBg, up);
                float glow = pow(max(0.0, 1.0 - length(uv) * 0.62), 3.0);
                c += ramp(0.45) * glow * (0.10 + ENERGY * 0.30);
                c += ramp(0.85) * PULSE * HARDEST * 0.05;
                return c;
            }

            void main() {
                // Normalised by the short side: the world is sized against the
                // dimension that is scarce, so a rotation or a different device
                // shows more of the scene instead of cropping into it.
                vec2 uv = (2.0 * gl_FragCoord.xy - uRes) / min(uRes.x, uRes.y);

                vec3 ro, ta;
                float far, stepScale;
                if (SCENE == 1) {
                    // Flying, not orbiting. Speed follows the tempo so the
                    // tunnel travels at the rate of the music.
                    float fly = uTime * (1.1 + TEMPO * 1.5 + ENERGY * 0.8);
                    ro = vec3(0.0, 0.0, fly);
                    ta = ro + vec3(0.0, 0.0, 1.0);
                    far = 26.0; stepScale = 0.75;
                } else if (SCENE == 3) {
                    float fly = uTime * 0.55;
                    ro = vec3(sin(uTime * 0.21) * 1.2, 0.4 + sin(uTime * 0.13) * 0.4, fly);
                    ta = ro + vec3(sin(uTime * 0.17) * 0.35, -0.12, 1.0);
                    far = 22.0; stepScale = 0.80;
                } else {
                    float az = uTime * 0.16 + 0.6;
                    float rad = 3.05 - BASS * 0.35 - PULSE * 0.12;
                    ro = vec3(sin(az) * rad, 0.55 + sin(uTime * 0.23) * 0.28, cos(az) * rad);
                    ta = vec3(0.0, 0.0, 0.0);
                    far = 12.0; stepScale = 0.70;
                }

                vec3 fwd = normalize(ta - ro);
                vec3 rgt = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
                vec3 upv = cross(rgt, fwd);
                vec3 rd = normalize(uv.x * rgt + uv.y * upv + 1.45 * fwd);

                float t = 0.02;
                float mat = 0.0;
                int steps = 0;
                bool hit = false;
                for (int i = 0; i < MAX_STEPS; i++) {
                    vec3 pos = ro + rd * t;
                    vec2 h = map(pos);
                    // Ridged and domain-folded fields overstate how far it is
                    // safe to travel, so the step is shortened rather than the
                    // detail thrown away. Undershooting costs iterations;
                    // overshooting punches holes through the surface.
                    if (h.x < 0.0012 * t + 0.0009) { mat = h.y; hit = true; break; }
                    t += h.x * stepScale;
                    steps = i;
                    if (t > far) break;
                }

                vec3 col;
                if (hit) {
                    vec3 pos = ro + rd * t;
                    vec3 n = normalAt(pos);
                    vec3 l1 = normalize(vec3(0.55, 0.80, -0.35));
                    vec3 l2 = normalize(vec3(-0.60, 0.25, 0.60));
                    float dif = max(dot(n, l1), 0.0);
                    float fill = max(dot(n, l2), 0.0) * 0.35;
                    float spec = pow(max(dot(reflect(-l1, n), -rd), 0.0), 30.0);
                    float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
                    // Iterations stand in for occlusion: a ray that had to
                    // creep took many small steps, and creeping happens in
                    // crevices. Free, and it lands in the right places.
                    float ao = clamp(1.0 - float(steps) / float(MAX_STEPS) * 1.5, 0.15, 1.0);
                    vec3 base = ramp(mat);
                    col = base * (0.16 + 0.74 * dif + fill) * ao;
                    col += ramp(0.95) * spec * (0.35 + TREBLE * 0.5);
                    col += ramp(0.70) * rim * (0.22 + PULSE * 0.35);
                    // Distance fades into the sky rather than into black, so
                    // the far end of a tunnel is part of the same picture.
                    col = mix(col, sky(uv, rd), clamp(t / far, 0.0, 1.0) * 0.9);
                } else {
                    col = sky(uv, rd);
                }

                // Vignette and a filmic knee: without the roll-off the specular
                // highlights clip to white and lose the theme's hue exactly
                // where the eye is looking.
                col *= 1.0 - 0.25 * pow(length(uv) * 0.5, 2.5);
                col = col / (col + vec3(0.85)) * 1.6;
                fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.90)), 1.0);
            }
        """.trimIndent()
    }
}
