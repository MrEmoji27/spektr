<!--
Body of the GitHub release for v0.6.0. Publishing:

    gh release create v0.6.0 --draft --title "spektr 0.6.0" \
      --notes-file docs/release-notes-0.6.0.md
    git tag -a v0.6.0 -m "spektr 0.6.0" && git push origin v0.6.0
    # the tag also runs publish-pypi.yml: set up trusted publishing on PyPI
    # first (owner MrEmoji27, repository spektr, workflow publish-pypi.yml)
    # once the exe, installer, Linux binary and APK are attached:
    gh release edit v0.6.0 --draft=false
-->

spektr now listens to the music, not just how loud it is. It can tell the
drums apart, feel where each bar begins, and follow the chords, and the
visuals react to the big moments of a song instead of every small sound.

**69 visuals (9 new) · 55 colour themes**

### It follows the song

- **Reacts to the hits that matter.** The main beats and hits move the
  picture. Quiet background sounds no longer set it off.
- **Hears the drums.** It can tell a kick drum from a snare from a hi-hat.
- **Feels the bar.** It knows where each bar of music begins, and says so
  when it can't tell.
- **Follows the chords.** It hears the notes being played and works out the
  key of the song.
- **Fast songs, right speed.** Songs above 150 beats a minute were read at
  half speed. Not any more.
- **Smooth, but not late.** The smooth motion setting still lets the big hits
  punch through straight away.

### Nine new visuals

| Visual | What you see |
|---|---|
| **JP Sequencer** | a grid of glowing pads, each drum playing its own light show |
| **JP Chords** | the chord you're hearing, in big letters on a synth display |
| **JP Panel** | a drum machine's pads lighting up where each drum lands |
| **JP Tracker** | the song scrolling past: notes on top, drums below |
| **Riptide** | two sets of waves crossing each other, sparking where they meet |
| **Twin Storms** | two whirlpools spinning against each other |
| **Arc Storm** | lightning across the screen, forking on every hit |
| **Shatter** | a pane of glass that breaks where each hit lands |
| **Locket Beat** | a heart sending out a steady stream of rings |

The older JP meters got a refresh too: colour zones like a car stereo, peak
lights that hold and drop, and a flash on every beat.

### Better on the beat

- **Fireworks** now launch just before each beat and burst right on it.
- **Pulse** and **Shooting Star** hit harder on the beat.
- **Switching visuals** looks cleaner: each kind arrives its own way (bars
  rise, sparks burst, rings ripple) and the old picture fades out. It is also
  faster. The old style is still in settings as "classic".

### Lighter on your computer

- Uses **about 90% less memory** at start (45 MB, down from 527 MB).
- Uses less processor time, and drops to a slow idle when nothing is playing.
- The first switch into a visual no longer stutters.

### Easier to install and use

- **One line to install**, on Windows, Linux and macOS. On Windows it adds
  spektr to the Start Menu, needs no admin rights, and comes with an
  uninstaller.
- **Also on PyPI**, for people who use Python: `pip install spektr-audio`.
- **A clearer settings panel.** One line per setting, and a plain explanation
  of the one you're on.
- **A clearer help screen** (press `h`): what every key does, and your
  current settings.
- **Friendlier typing mistakes.** Mistype a visual's name and spektr suggests
  the right one.

For people who build their own visuals: the new sounds spektr can hear (drums,
bar, chords, key) are available to plugins too.

### Not yet

- Spotting build-ups and drops is built but switched off until it has been
  tested on more music.
- Saving a picture of the screen, themes from album art, and recovering on
  its own when no audio device is found are still to come.

### Next

0.6.5 brings spektr to macOS properly, with no extra audio setup needed.

### Which download

| You have | Do this |
|---|---|
| Windows | paste into PowerShell: `irm https://raw.githubusercontent.com/MrEmoji27/spektr/main/install.ps1 \| iex` |
| Windows, no install | download `spektr.exe` and double-click it |
| Windows, classic installer | `spektr-0.6.0.0-setup.exe` |
| Linux or macOS | paste into a terminal: `curl -fsSL https://raw.githubusercontent.com/MrEmoji27/spektr/main/install.sh \| sh` |
| Python | `pip install spektr-audio` |
| Android | the `.apk` file, Android 10 or newer |

Windows may warn the first time because the app isn't signed: choose **More
info**, then **Run anyway**.
