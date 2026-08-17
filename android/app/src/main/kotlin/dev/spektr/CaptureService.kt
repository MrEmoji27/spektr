package dev.spektr

import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.drawable.Icon
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Foreground service doing AudioPlaybackCapture and feeding EngineManager.
 *
 * Flow per the design: MediaProjection consent happens in MainActivity; the
 * result token travels here as extras; on start we enter the foreground FIRST
 * (Android 14 requires a running mediaProjection FGS before
 * getMediaProjection), then build the capture AudioRecord and read float PCM
 * into the engine's ring buffer until the projection dies or Stop is pressed.
 *
 * A blocked source is not an error here: the app captures whatever the
 * system gives it, and silence reaches the engine like any other audio.
 * Naming the blocking app is v3 (notification-listener grant).
 */
class CaptureService : Service() {

    private val running = AtomicBoolean(false)
    private var record: AudioRecord? = null
    private var projection: MediaProjection? = null
    private var readThread: Thread? = null

    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            Log.i(TAG, "projection revoked by the system")
            stopSelf()
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (running.get()) return START_NOT_STICKY

        enterForeground()

        val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
            ?: Activity.RESULT_CANCELED
        @Suppress("DEPRECATION")
        val data = intent?.getParcelableExtra<Intent>(EXTRA_RESULT_DATA)
        if (resultCode != Activity.RESULT_OK || data == null) {
            Log.e(TAG, "no consent token, refusing to capture")
            stopSelf()
            return START_NOT_STICKY
        }
        startCapture(resultCode, data)
        return START_NOT_STICKY
    }

    private fun enterForeground() {
        val channelId = CHANNEL_ID
        if (Build.VERSION.SDK_INT >= 26) {
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(
                NotificationChannel(channelId, "capture", NotificationManager.IMPORTANCE_LOW)
            )
        }
        val stopPi = PendingIntent.getService(
            this, 0, Intent(this, CaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE
        )
        val openPi = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )
        val notification = Notification.Builder(this, channelId)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(getString(R.string.notification_capturing))
            .setContentIntent(openPi)
            .setOngoing(true)
            .addAction(
                Notification.Action.Builder(
                    Icon.createWithResource(this, R.drawable.ic_notification),
                    getString(R.string.notification_stop),
                    stopPi,
                ).build()
            )
            .build()
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIF_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        } else {
            startForeground(NOTIF_ID, notification)
        }
    }

    private fun startCapture(resultCode: Int, data: Intent) {
        try {
            val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            val proj = mpm.getMediaProjection(resultCode, data)
            projection = proj
            proj.registerCallback(projectionCallback, Handler(Looper.getMainLooper()))

            val config = AudioPlaybackCaptureConfiguration.Builder(proj)
                .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
                .addMatchingUsage(AudioAttributes.USAGE_GAME)
                .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
                .build()

            val format = AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(SAMPLE_RATE)
                .setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
                .build()

            var minBytes = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_STEREO, AudioFormat.ENCODING_PCM_FLOAT
            )
            if (minBytes <= 0) minBytes = BUCKET_FLOATS * 4 * 2
            val rec = AudioRecord.Builder()
                .setAudioFormat(format)
                .setAudioPlaybackCaptureConfig(config)
                .setBufferSizeInBytes(minBytes)
                .build()
            rec.startRecording()
            record = rec

            running.set(true)
            CaptureController.state = CaptureController.State.Capturing
            EngineManager.syncRendering()
            readThread = Thread({ readLoop(rec) }, "spektr-capture").also { it.start() }
            Log.i(TAG, "capture running at ${SAMPLE_RATE} Hz float stereo")
        } catch (t: Throwable) {
            Log.e(TAG, "capture failed to start", t)
            stopSelf()
        }
    }

    private fun readLoop(rec: AudioRecord) {
        val bucket = FloatArray(BUCKET_FLOATS)
        val bytes = ByteBuffer.allocate(BUCKET_FLOATS * 4).order(ByteOrder.LITTLE_ENDIAN)
        while (running.get()) {
            val n = rec.read(bucket, 0, bucket.size, AudioRecord.READ_BLOCKING)
            if (n <= 0) {
                Log.w(TAG, "read returned $n")
                continue
            }
            bytes.clear()
            bytes.asFloatBuffer().put(bucket, 0, n)
            bytes.limit(n * 4)
            EngineManager.push(bytes.array().copyOf(n * 4))
        }
    }

    override fun onDestroy() {
        running.set(false)
        readThread?.interrupt()
        readThread = null
        record?.let {
            runCatching { it.stop() }
            it.release()
        }
        record = null
        projection?.let {
            runCatching { it.unregisterCallback(projectionCallback) }
            it.stop()
        }
        projection = null
        CaptureController.state = CaptureController.State.Idle
        // Not an unconditional stop: if the app is still on screen it goes on
        // previewing the mode, and only a backgrounded app with no capture
        // has nothing left to draw for.
        EngineManager.syncRendering()
        Log.i(TAG, "capture stopped")
        super.onDestroy()
    }

    companion object {
        private const val TAG = "spektr"
        private const val CHANNEL_ID = "capture"
        private const val NOTIF_ID = 1
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        private const val ACTION_STOP = "dev.spektr.action.STOP"

        const val SAMPLE_RATE = 48000
        private const val BUCKET_FLOATS = 4096
    }
}
