package dev.spektr

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat

/**
 * Consent -> capture -> grid.
 *
 * The activity owns only the permission dance and the projection consent;
 * mode and theme live in [EngineManager] and are picked from the screen, so
 * nothing about a switch comes back through here.
 */
class MainActivity : ComponentActivity() {

    private val consentLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            startCaptureService(result.resultCode, result.data!!)
        }
    }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ ->
        // The foreground service runs regardless; the grant only controls
        // whether its notification is visible. Consent opens either way.
        launchConsent()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        EngineManager.start(applicationContext)
        setContent {
            SpektrScreen(
                onStartCapture = ::onStartCaptureClicked,
                onStopCapture = ::stopCapture,
            )
        }
    }

    // The home screen previews the selected mode, so frames are wanted
    // whenever this window is up — and only then, unless capture is running.
    override fun onResume() {
        super.onResume()
        EngineManager.setUiVisible(true)
    }

    override fun onPause() {
        super.onPause()
        EngineManager.setUiVisible(false)
    }

    private val audioPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        // Not optional, unlike the notification one: denied, AudioRecord
        // cannot be built at all, so opening the consent dialog would ask for
        // a screen we are never going to read.
        if (granted) askNotificationsThenConsent() else EngineManager.reportNoAudioPermission()
    }

    /**
     * RECORD_AUDIO first, then notifications, then the projection consent.
     *
     * The audio permission is easy to leave out — the projection token is what
     * everyone associates with capture, and it genuinely is all you need to
     * capture the *screen*. Audio is different: an AudioRecord built from an
     * AudioPlaybackCaptureConfiguration is still an audio record, and the
     * platform refuses it without RECORD_AUDIO. Leave it out and the app
     * starts, draws, and shows a picture of silence with nothing to say why.
     */
    private fun onStartCaptureClicked() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            return
        }
        askNotificationsThenConsent()
    }

    private fun askNotificationsThenConsent() {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            launchConsent()
        }
    }

    private fun launchConsent() {
        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        consentLauncher.launch(mpm.createScreenCaptureIntent())
    }

    private fun startCaptureService(resultCode: Int, data: Intent) {
        CaptureController.state = CaptureController.State.Capturing
        val intent = Intent(this, CaptureService::class.java)
            .putExtra(CaptureService.EXTRA_RESULT_CODE, resultCode)
            .putExtra(CaptureService.EXTRA_RESULT_DATA, data)
        ContextCompat.startForegroundService(this, intent)
    }

    private fun stopCapture() {
        stopService(Intent(this, CaptureService::class.java))
    }
}
