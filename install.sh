#!/bin/sh
# spektr installer for Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/MrEmoji27/spektr/main/install.sh | sh
#
# Linux on x86-64 gets the release binary in ~/.local/bin, no Python needed.
# macOS and other machines get spektr from PyPI through uv, which brings its
# own Python. Run it again to update. No sudo.
#
# Settings, as environment variables, all optional:
#   SPEKTR_VERSION   a release to install, like v0.6.0 (default: the latest)
#   SPEKTR_DIR       where the binary goes (default: ~/.local/bin)
#   SPEKTR_FROM      install this local spektr binary instead of downloading
#   SPEKTR_USE_UV    install from PyPI with uv even where a binary exists
#   NO_COLOR         plain text

set -eu

REPO="MrEmoji27/spektr"
DIR="${SPEKTR_DIR:-$HOME/.local/bin}"

# -- output --------------------------------------------------------------------

LIVE=0
[ -t 1 ] && LIVE=1
COLOR=0
if [ "$LIVE" = 1 ] && [ -z "${NO_COLOR:-}" ]; then COLOR=1; fi
E=$(printf '\033')

# the spektr sunset, violet to gold, as "r g b" at 0..1
paint() {  # paint <0..1> <text>
    if [ "$COLOR" = 1 ]; then
        awk -v u="$1" -v s="$2" -v e="$E" 'BEGIN {
            split("123 47 247,241 7 163,255 109 0,255 208 0", st, ",")
            if (u < 0) u = 0; if (u > 1) u = 1
            x = u * 3; i = int(x); if (i > 2) i = 2; f = x - i
            split(st[i + 1], a, " "); split(st[i + 2], b, " ")
            printf "%s[38;2;%d;%d;%dm%s%s[0m", e, a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f, s, e }'
    else
        printf '%s' "$2"
    fi
}
dim() { if [ "$COLOR" = 1 ]; then printf '%s[2m%s%s[0m' "$E" "$1" "$E"; else printf '%s' "$1"; fi; }
bold() { if [ "$COLOR" = 1 ]; then printf '%s[1m%s%s[0m' "$E" "$1" "$E"; else printf '%s' "$1"; fi; }
green() { if [ "$COLOR" = 1 ]; then printf '%s[38;2;80;220;120m%s%s[0m' "$E" "$1" "$E"; else printf '%s' "$1"; fi; }
red() { if [ "$COLOR" = 1 ]; then printf '%s[38;2;255;80;80m%s%s[0m' "$E" "$1" "$E"; else printf '%s' "$1"; fi; }

STEP=""
begin() { STEP="$1"; [ "$LIVE" = 1 ] && printf '  %s %-24s' "$(paint 0.66 '▸')" "$STEP"; return 0; }
done_() { printf '\r  %s %-24s%s\n' "$(green '✓')" "$STEP" "$(dim "$1")"; }
fail() {
    printf '\n  %s %s\n' "$(red '✗')" "$(bold "$1")"
    [ -n "${2:-}" ] && printf '    %s\n' "$2"
    printf '\n'
    exit 1
}

# One frame of a spectrum: <width> bars moving with <t>, the first <lit> of
# them lit and the rest dots.
spectrum() {
    awk -v w="$1" -v t="$2" -v lit="$3" -v color="$COLOR" -v e="$E" 'BEGIN {
        split("▁ ▂ ▃ ▄ ▅ ▆ ▇ █", g, " ")
        split("123 47 247,241 7 163,255 109 0,255 208 0", st, ",")
        for (i = 0; i < w; i++) {
            if (i >= lit) { printf (color ? e "[2m·" e "[0m" : "·"); continue }
            h = 0.5 + 0.5 * sin(t * 7 + i * 0.55) * cos(t * 3.1 - i * 0.23)
            c = g[int(h * 7 + 0.5) + 1]
            if (!color) { printf "%s", c; continue }
            u = (w > 1) ? i / (w - 1) : 0; x = u * 3; k = int(x); if (k > 2) k = 2; f = x - k
            split(st[k + 1], a, " "); split(st[k + 2], b, " ")
            printf "%s[38;2;%d;%d;%dm%s%s[0m", e, a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f, c, e
        } }'
}

# fractional seconds where the shell has them, whole ones where it does not
clock() { date +%s.%N 2>/dev/null | grep -v N || date +%s; }

# -- the banner --------------------------------------------------------------------

printf '\n'
for row in \
    ' ####  #####  ###### #    # ###### ##### ' \
    '#      #    # #      #   #    #    #    #' \
    ' ####  #####  #####  ####     #    ##### ' \
    '     # #      #      #   #    #    #  #  ' \
    ' ####  #      ###### #    #   #    #   # '
do
    printf '  '
    if [ "$COLOR" = 1 ]; then
        printf '%s' "$row" | awk -v e="$E" '{
            split("123 47 247,241 7 163,255 109 0,255 208 0", st, ",")
            n = length($0)
            for (i = 1; i <= n; i++) {
                ch = substr($0, i, 1)
                if (ch != "#") { printf " "; continue }
                u = (i - 1) / (n - 1); x = u * 3; k = int(x); if (k > 2) k = 2; f = x - k
                split(st[k + 1], a, " "); split(st[k + 2], b, " ")
                printf "%s[38;2;%d;%d;%dm█%s[0m", e, a[1] + (b[1] - a[1]) * f,
                    a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f, e
            } }'
    else
        printf '%s' "$row" | sed 's/#/█/g'
    fi
    printf '\n'
done
printf '  %s\n\n' "$(dim 'a music visualiser for your terminal')"
if [ "$LIVE" = 1 ]; then
    i=0
    while [ $i -lt 22 ]; do
        printf '\r  %s' "$(spectrum 41 "$(awk -v i=$i 'BEGIN { print i * 0.05 }')" 41)"
        sleep 0.05 2>/dev/null || sleep 1
        i=$((i + 1))
    done
    printf '\r  %41s\r' ''
fi

# -- 1. the system -------------------------------------------------------------------

begin 'Checking your system'
OS=$(uname -s)
ARCH=$(uname -m)
HOW=uv
if [ -n "${SPEKTR_FROM:-}" ]; then
    HOW=local
elif [ "$OS" = Linux ] && { [ "$ARCH" = x86_64 ] || [ "$ARCH" = amd64 ]; } && [ -z "${SPEKTR_USE_UV:-}" ]; then
    HOW=binary
fi
case "$OS" in
    Linux)  NAME="Linux" ;;
    Darwin) NAME="macOS" ;;
    *)      NAME="$OS" ;;
esac
case "$HOW" in
    binary) done_ "$NAME ($ARCH), the release binary" ;;
    local)  done_ "$NAME ($ARCH), a local build" ;;
    uv)     done_ "$NAME ($ARCH), from PyPI through uv" ;;
esac
command -v curl >/dev/null 2>&1 || fail 'spektr needs curl to install.' 'Install curl, then run this again.'

# -- 2a. the release binary ----------------------------------------------------------

install_binary() {
    tmp=$(mktemp)
    if [ "$HOW" = local ]; then
        begin 'Using a local build'
        [ -f "$SPEKTR_FROM" ] || fail "There is no file at $SPEKTR_FROM."
        cp "$SPEKTR_FROM" "$tmp"
        done_ "$SPEKTR_FROM"
    else
        tag="${SPEKTR_VERSION:-}"
        begin 'Finding the release'
        if [ -z "$tag" ]; then
            tag=$(curl -fsSL -H 'User-Agent: spektr-installer' \
                "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null |
                sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1) || tag=""
        fi
        if [ -n "$tag" ]; then
            url="https://github.com/$REPO/releases/download/$tag/spektr"
            done_ "$tag"
        else
            url="https://github.com/$REPO/releases/latest/download/spektr"
            done_ 'the latest'
        fi

        begin 'Downloading spektr'
        total=$(curl -fsSIL "$url" 2>/dev/null | tr -d '\r' |
            awk 'tolower($1) == "content-length:" { n = $2 } END { print n + 0 }')
        curl -fsSL "$url" -o "$tmp" &
        pid=$!
        start=$(clock)
        if [ "$LIVE" = 1 ]; then
            while kill -0 "$pid" 2>/dev/null; do
                got=$(wc -c < "$tmp" | tr -d ' ')
                t=$(awk -v a="$start" -v b="$(clock)" 'BEGIN { print b - a }')
                lit=$(awk -v g="$got" -v n="$total" 'BEGIN { print (n > 0) ? int(28 * g / n) : 0 }')
                info=$(awk -v g="$got" -v n="$total" -v t="$t" 'BEGIN {
                    s = sprintf("%.1f", g / 1048576)
                    if (n > 0) s = s sprintf(" / %.1f MB", n / 1048576); else s = s " MB"
                    if (t > 0) s = s sprintf("  %.1f MB/s", g / 1048576 / t)
                    print s }')
                printf '\r  %s %-24s%s %s   ' "$(paint 0.66 '▸')" "$STEP" "$(spectrum 28 "$t" "$lit")" "$(dim "$info")"
                sleep 0.08 2>/dev/null || sleep 1
            done
        fi
        wait "$pid" || { rm -f "$tmp"; fail 'The download failed.' "Check the release exists: https://github.com/$REPO/releases"; }
        [ "$LIVE" = 1 ] && printf '\r%90s\r' ''
        done_ "$(awk -v g="$(wc -c < "$tmp")" 'BEGIN { printf "%.1f MB", g / 1048576 }')"
    fi

    # a Linux program starts with ELF; anything else is an error page
    magic=$(head -c 4 "$tmp" | od -An -c | tr -d ' ')
    size=$(wc -c < "$tmp" | tr -d ' ')
    if [ "$HOW" = binary ] && { [ "$size" -lt 1048576 ] || [ "$magic" != '177ELF' ]; }; then
        rm -f "$tmp"
        fail 'What came down is not spektr.' "Try again, or download it from https://github.com/$REPO/releases"
    fi

    begin 'Installing'
    mkdir -p "$DIR"
    chmod +x "$tmp"
    mv -f "$tmp" "$DIR/spektr"
    done_ "$DIR/spektr"

    if [ "$HOW" = binary ]; then
        # the audio libraries the binary loads at run time
        begin 'Checking audio libraries'
        missing=""
        have() { ldconfig -p 2>/dev/null | grep -q "$1" || ls /usr/lib*/"$1"* /usr/lib/*/"$1"* >/dev/null 2>&1; }
        have libportaudio || missing="PortAudio"
        have libpulse || missing="${missing:+$missing and }libpulse"
        if [ -z "$missing" ]; then
            done_ 'PortAudio and libpulse are there'
        else
            id=$(. /etc/os-release 2>/dev/null && echo "${ID:-}")
            case "$id" in
                arch|manjaro|endeavouros) hint='sudo pacman -S portaudio libpulse' ;;
                debian|ubuntu|linuxmint|pop) hint='sudo apt install libportaudio2 libpulse0' ;;
                fedora) hint='sudo dnf install portaudio pulseaudio-libs' ;;
                *) hint='install PortAudio and libpulse from your package manager' ;;
            esac
            done_ "missing $missing: $hint"
        fi
    fi
    SPEKTR="$DIR/spektr"
}

# -- 2b. from PyPI through uv -----------------------------------------------------------

install_uv() {
    begin 'Finding uv'
    if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
        printf '\r'
        begin 'Installing uv'
        curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 ||
            fail 'uv would not install.' 'See https://docs.astral.sh/uv/ and run this again.'
    fi
    UV=$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")
    done_ "$("$UV" --version 2>/dev/null || echo uv)"

    begin 'Installing spektr'
    pkg="spektr-audio"
    [ -n "${SPEKTR_VERSION:-}" ] && pkg="spektr-audio==${SPEKTR_VERSION#v}"
    log=$(mktemp)
    "$UV" tool install --force --python 3.12 "$pkg" > "$log" 2>&1 &
    pid=$!
    start=$(clock)
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$LIVE" = 1 ]; then
            t=$(awk -v a="$start" -v b="$(clock)" 'BEGIN { print b - a }')
            printf '\r  %s %-24s%s' "$(paint 0.66 '▸')" "$STEP" "$(spectrum 16 "$t" 16)"
        fi
        sleep 0.08 2>/dev/null || sleep 1
    done
    if ! wait "$pid"; then
        printf '\n'; tail -n 5 "$log"; rm -f "$log"
        fail 'spektr would not install.' "Try it yourself: uv tool install $pkg"
    fi
    rm -f "$log"
    [ "$LIVE" = 1 ] && printf '\r%60s\r' ''
    "$UV" tool update-shell >/dev/null 2>&1 || true
    done_ "$pkg"
    SPEKTR=$(command -v spektr 2>/dev/null || echo "$HOME/.local/bin/spektr")
}

case "$HOW" in
    uv) install_uv ;;
    *)  install_binary ;;
esac

# -- 3. check it runs -------------------------------------------------------------------

begin 'Checking it starts'
version=$("$SPEKTR" --version 2>/dev/null) || fail 'spektr is installed but would not start.' "Try running it yourself: $SPEKTR"
done_ "$version"
begin 'Loading every mode'
modes=$("$SPEKTR" --check-modes 2>/dev/null) || fail "Some of spektr's modes failed to load." "Run $SPEKTR --check-modes to see which."
done_ "$modes"

# -- done ----------------------------------------------------------------------------

path_note=""
case ":$PATH:" in
    *":$(dirname "$SPEKTR"):"*) ;;
    *) path_note="Add $(dirname "$SPEKTR") to your PATH, or open a new terminal." ;;
esac
# the box sizes itself to what is in it
set -- "$version is installed." "" "Start it       spektr" "Every option   spektr --help"
if [ "$HOW" = uv ]; then set -- "$@" "Uninstall      uv tool uninstall spektr-audio"
else set -- "$@" "Uninstall      rm $SPEKTR"; fi
[ -n "$path_note" ] && set -- "$@" "" "$path_note"
wide=0
for l in "$@"; do [ ${#l} -gt "$wide" ] && wide=${#l}; done
rule=$(awk -v n=$((wide + 4)) 'BEGIN { for (i = 0; i < n; i++) printf "─" }')
printf '\n  %s\n' "$(paint 0.1 "╭$rule╮")"
first=1
for l in "$@"; do
    text="$l"
    if [ "$first" = 1 ]; then text="$(bold "$l")"; first=0; fi
    printf '  %s  %s%*s  %s\n' "$(paint 0.1 '│')" "$text" $((wide - ${#l})) '' "$(paint 0.9 '│')"
done
printf '  %s\n\n' "$(paint 0.9 "╰$rule╯")"
