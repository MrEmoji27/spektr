<!--
Draft body of the GitHub release for v0.5.5, updated as each piece lands.
Items marked "planned" are not finished yet. Publishing:

    gh release create v0.5.5 --draft --title "spektr 0.5.5" \
      --notes-file docs/release-notes-0.5.5.md
    git tag -a v0.5.5 -m "spektr 0.5.5" && git push origin v0.5.5
    # once the exe, installer, Linux binary and APK are attached:
    gh release edit v0.5.5 --draft=false
-->

> **Still being built.** Anything marked *planned* is not finished yet.

No new modes this time, and no new themes. 0.5.5 is the release that makes
spektr lighter, smoother and harder to break — the groundwork for what comes
next.

**74 modes, 55 themes, both unchanged.**

### Lighter on old machines — *planned*

spektr asked more of your processor than it had any right to, and most of it
went on drawing the frame rather than on the visual itself. This release goes
after that, aiming to roughly a third of what it uses today.

There is also a new **eco** setting: 30 frames a second, fewer bars, and
shuffle keeping away from the heaviest visuals. It switches itself on when
your machine can't keep up, or you can turn it on yourself.

### Shuffle fades instead of cutting — *planned*

Shuffle used to jump straight to the next visual. Now the old picture
dissolves into the new one over about half a second, dot by dot. Pressing `m`
or picking from the list still switches instantly — a key press should feel
immediate.

You can also have shuffle wait for the song to change instead of watching a
clock.

### Linux: it starts on older systems again — *planned*

The Linux binary was built on too new a system and refused to start on older
ones with a `GLIBC` error. It's now built for **glibc 2.28** — Ubuntu 18.10,
Debian 10, Fedora 29 and anything newer — and every release is started on
Ubuntu 20.04 before it ships, so this can't quietly come back.

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

Behind that, the project is being reorganised into clear parts, and memory is
capped: only the five most recently used visuals stay loaded. *Planned.*

### For plugin authors

Nothing you use changes: `spektr.api` keeps every name, and the mode contract
and plugin version stay as they are. If you import from somewhere inside
spektr instead of `spektr.api`, that still works here, but move across — those
paths go away in 0.6.5.

### Next

0.6.0 is about hearing the music rather than its volume: kick, snare and hat
apart from each other, where the bar begins, build-ups and drops, colour that
follows the key — with new JP meters built on top, and spektr on PyPI so `pip`
can install it.
