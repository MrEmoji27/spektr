<!--
The body of the GitHub release for v0.4.0.

It has to exist *before* the tag is pushed. All three build workflows attach
their artefacts with `append_body: true`, which means they add a download
section to whatever is already there — they do not write the notes. Push the
tag with an empty release and the published page is three download blurbs and
nothing about what changed.

    gh release create v0.4.0 --draft --title "spektr 0.4.0 — the rhythm release" \
      --notes-file docs/release-notes-0.4.0.md

Then push the tag, let the three builds attach their files, and publish.
-->

**spektr knows where the beat is now.** That is the release, and most of the
rest of it is something that made possible.

Everything that reacted to rhythm before was really reacting to bass energy,
because that is what was available — and a kick and a loud sustained low note
look identical to a level meter. There is a real onset detector under the app
now: spectral flux, adaptive thresholding, per-sub-band peak picking, region
gates, and a rescue arm for hits masked by the drum before them.

It is scored rather than asserted. Against an 11-scenario corpus written
against the problem instead of the implementation, at ±50 ms with one-to-one
matching: **precision 1.000, recall 0.940, F 0.969**, and the build fails if
that drops. Precision is the number that matters most here — a visualiser
that flashes on a beat that did not happen is worse than one that misses a
quiet one.

**Terminal cells can hold more than one colour now.** The Unicode octant block
gives eight lit regions in two colours per cell, so twelve modes have an `(o)`
variant that draws a curve as a stroke instead of a staircase. Fonts are
patchy about it, so it is opt-in, `spektr --glyph-test` shows what yours has,
and there is a quadrant fallback for fonts with no octants at all.

**There is an Android build.** Same engine, same modes, same themes, running
on a tablet as an ambient display for the desk. It lives in this repository
under `android/` and the APK is attached below. It is a second screen, not a
second product — it draws what *that device* is playing, because Android has
no way for one device to read another's audio.

**Nine new modes** — Shooting Star, Snow, Valentine, Locket, Kaleidoscope,
Tunnel In, Dither, Dither Storm, Dither Storm Extreme — plus the twelve `(o)`
variants. Flipbook is gone. **Six new themes**: emerald, sapphire, amethyst,
citrine, tangerine, indigo.

**44 modes → 65. 49 themes → 55.**

Ten modes were rebuilt around what was actually costing rather than what
looked expensive, `make_strips` encodes the whole grid in one pass, and the
benchmark can now fail — it had printed every number needed to catch a mode
sitting at 10.8 ms against a 16.7 ms budget for the whole of the project's
life, and nothing read them.

Also fixed: the analysis hop rate was tied to the capture block size, so
changing your audio device changed the rhythm reading. Non-finite samples
reached the FFT. `d` leaked a thread on every press. The goniometer's geometry
was wrong. A 1-D stereo buffer crashed the scope modes.

[Full changelog](https://github.com/MrEmoji27/spektr/blob/main/CHANGELOG.md)

---

### Which file do I want?

| you have | download |
|---|---|
| Windows, no Python | `spektr.exe` — portable, double-click |
| Windows, want a Start Menu entry | `spektr-0.4.0.0-setup.exe` |
| Linux, no Python | `spektr` — `chmod +x` and run |
| Android tablet or phone | `spektr-android-0.2.0-arm64-v8a.apk` |
| Python already | clone the repo and `pip install -e .` — spektr is not on PyPI yet |

The Android APK carries its own version line — **0.2.0**, because that is how
many versions of the port there have been — and ships inside this release.

The Windows and Linux builds are unsigned, because certificates cost money.
SmartScreen will warn on first run: **More info → Run anyway**.
