<!--
Body of the GitHub release for v0.5.5. Publishing:

    gh release create v0.5.5 --draft --title "spektr 0.5.5" \
      --notes-file docs/release-notes-0.5.5.md
    git tag -a v0.5.5 -m "spektr 0.5.5" && git push origin v0.5.5
    # once the exe, installer, Linux binary and APK are attached:
    gh release edit v0.5.5 --draft=false
-->

No new modes this time, and no new themes. 0.5.5 is the release that makes
spektr lighter, smoother and harder to break — the groundwork for what comes
next.

**74 modes, 55 themes, both unchanged.**

### Eco mode, for machines that can't keep up

A new **eco** setting in the settings panel: 30 frames a second, fewer bars,
and shuffle staying away from the visuals your machine has been measured
struggling with. It is off until you turn it on — spektr does not decide this
for you.

Which visuals it avoids is measured here, on your machine, at your window
size, rather than read from a list: the same visual can be cheap in a small
window and far too slow fullscreen.

Honest about the rest: most of spektr's processor time goes on pushing frames
through the interface framework, not on drawing the visuals, and that has not
changed in this release. Measured here, the heaviest visual at a fullscreen
size still costs more than a frame's worth of time. The groundwork is done and
measured — see the note at the end — and the work itself is 0.6.x.

### Shuffle fades instead of cutting

Shuffle used to jump straight to the next visual. Now the old picture
dissolves into the new one over about half a second, dot by dot. Pressing `m`
or picking from the list still switches instantly — a key press should feel
immediate.

You can also have shuffle wait for the song to change instead of watching a
clock. The settings panel says when the computer cannot provide now-playing
information, so that option never fails silently.

### Linux: it starts on older systems again

The Linux binary was built on too new a system and refused to start on older
ones with a `GLIBC` error. It's now built for **glibc 2.28** — Ubuntu 18.10,
Debian 10, Fedora 29 and anything newer — and every release is started on
Ubuntu 20.04 before it ships, so this can't quietly come back.

### It hears the first beat now

A track's first hit used to be invisible. The detector compares each moment
with the one before it, and at the start of a track there is nothing before —
so the first kick after silence was never found, on every track, every time.
It is found now, and the same fix covers starting again after a pause.

spektr also used to assume every sound card runs at 48 kHz. On a 96 kHz device
every timing inside the beat detector meant half as long as it should, and
detection suffered for it: on the test corpus, accuracy rises from 0.888 to
0.905 and the hardest case — hi-hats close behind a drum — from 0.48 to 0.63.
At 44.1 and 48 kHz nothing changes.

### Your settings stay put

If you went back to an older spektr, it used to quietly erase any setting it
didn't recognise. It keeps them now.

And if you run the tests from a checkout, they no longer overwrite your own
settings — they used to, because the tests open the app, and the app saves
when it closes.

### Under the hood

Nothing here changes what you see, and that's checked rather than hoped for:
all 74 modes are recorded drawing a fixed piece of audio, and anything that
alters a picture fails the build. A 0.5.0 config folder — settings, loadouts,
a theme, a plugin — has to survive an upgrade untouched, and every flag and
key is pinned too.

Behind that, the project is now organised into clear parts. Audio capture,
analysis, modes, the interface and the command line live in focused packages
behind the same compatibility surface. Mode code loads only as it is needed,
and the running visualiser keeps a moving five-mode window — the current mode
and the next four in the same familiar order — rather than retaining state for
every mode visited in a long session.

### For plugin authors

Nothing you use changes: `spektr.api` keeps every name, and the mode contract
and plugin version stay as they are. If you import from somewhere inside
spektr instead of `spektr.api`, that still works here, but move across — those
paths go away in 0.6.5.

### One for the curious

The reason eco exists rather than a blanket speed-up: measured at a large
terminal size, building a frame costs about 1.4 ms and turning it into the
text a terminal understands costs 0.07 ms — while the whole app spends around
80% of a processor core. Nearly all of that goes on the interface framework
between those two steps. Replacing that path is the single biggest thing
spektr can do for old hardware, and it is too large a change to rush into a
release whose first rule is that nothing breaks.

### Next

0.6.0 is about hearing the music rather than its volume: kick, snare and hat
apart from each other, where the bar begins, build-ups and drops, colour that
follows the key — with new JP meters built on top, and spektr on PyPI so `pip`
can install it.

### Which file

| you have | download |
|---|---|
| Windows, no Python | `spektr.exe`, portable, double-click |
| Windows, one line in PowerShell | `irm https://github.com/MrEmoji27/spektr/releases/latest/download/spektr.exe -OutFile spektr.exe; ./spektr.exe` |
| Windows, want a Start Menu entry | `spektr-0.5.5.0-setup.exe` |
| Linux, no Python | `spektr`, `chmod +x` and run. Needs glibc 2.28 or newer |
| Android | `spektr-android-0.4.0-arm64-v8a.apk`, Android 10+, 64-bit ARM |
| Python already | clone and `pip install -e .`, not on PyPI yet |

The Windows and Linux builds are unsigned. SmartScreen will warn on first run:
More info, then Run anyway.
