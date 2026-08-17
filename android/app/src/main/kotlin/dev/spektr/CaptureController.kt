package dev.spektr

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/** Who owns the screen: nothing, or the capture foreground service. */
object CaptureController {
    enum class State { Idle, Capturing }

    var state by mutableStateOf(State.Idle)
}
