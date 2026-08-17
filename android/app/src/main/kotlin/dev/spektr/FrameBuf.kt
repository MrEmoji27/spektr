package dev.spektr

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * One rendered frame, parsed from the bridge's wire format.
 *
 * Layout (little-endian, no padding): a 12-byte header ("SPKT", version u16,
 * planes u16, w u16, h u16), then codes as w*h int32 (Unicode codepoints),
 * then cidx as w*h uint8 (foreground ramp index), then bidx as w*h uint8 —
 * only when planes == 3. The plane count must be honoured: dropping the
 * background plane of a half-block mode does not crash, it renders half-wrong.
 *
 * `planes == 1` is the field format: no codes and no glyphs, just w*h ramp
 * indices, one per picture pixel, with [FIELD_EMPTY] for "paint the
 * background". The grid is the terminal's constraint, not the mode's, and
 * this is the same frame with that constraint taken off.
 */
class FrameBuf(
    val w: Int,
    val h: Int,
    val planes: Int,
    val codes: IntArray,
    val cidx: ByteArray,
    val bidx: ByteArray?,
) {
    val size: Int get() = w * h

    /** True when this is a picture to blit rather than a grid to typeset. */
    val isField: Boolean get() = planes == 1

    companion object {
        const val MAGIC = "SPKT"
        const val VERSION = 1
        const val HEADER_SIZE = 12

        /** Ramp indices are 0..63, so 255 is free to mean "background here". */
        const val FIELD_EMPTY = 255

        /** Strict parser: returns null for any unrecognised or truncated buffer. */
        fun parse(data: ByteArray): FrameBuf? {
            if (data.size < HEADER_SIZE) return null
            val bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4)
            bb.get(magic)
            if (String(magic, Charsets.US_ASCII) != MAGIC) return null
            val version = bb.short.toInt() and 0xFFFF
            val planes = bb.short.toInt() and 0xFFFF
            val w = bb.short.toInt() and 0xFFFF
            val h = bb.short.toInt() and 0xFFFF
            if (version != VERSION) return null
            if (planes < 1 || planes > 3) return null
            if (w == 0 || h == 0) return null
            val cells = w * h
            if (planes == 1) {
                if (data.size != HEADER_SIZE + cells) return null
                val field = ByteArray(cells)
                bb.get(field)
                return FrameBuf(w, h, 1, IntArray(0), field, null)
            }
            val expected = HEADER_SIZE + cells * 4 + cells + if (planes == 3) cells else 0
            if (data.size != expected) return null
            val codes = IntArray(cells)
            for (i in 0 until cells) codes[i] = bb.int
            val cidx = ByteArray(cells)
            bb.get(cidx)
            val bidx = if (planes == 3) {
                ByteArray(cells).also { bb.get(it) }
            } else {
                null
            }
            return FrameBuf(w, h, planes, codes, cidx, bidx)
        }
    }
}
