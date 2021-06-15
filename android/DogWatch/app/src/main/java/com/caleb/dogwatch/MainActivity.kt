package com.caleb.dogwatch

import android.Manifest
import android.content.Context
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.media.MediaPlayer
import android.media.SoundPool
import android.net.ConnectivityManager
import android.net.wifi.WifiManager
import android.os.Build
import android.view.WindowManager
import androidx.core.app.ActivityCompat
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.View
import android.widget.*
import androidx.annotation.RequiresApi
import androidx.constraintlayout.helper.widget.Flow
import androidx.fragment.app.FragmentContainerView
import org.java_websocket.client.WebSocketClient
import org.java_websocket.handshake.ServerHandshake
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import java.lang.Exception
import java.math.RoundingMode
import java.net.*
import java.text.DecimalFormat

class MainActivity : AppCompatActivity(R.layout.activity_main) {

    private val PREDICTION_CUTOFF = 0.85
    private val PERMISSIONS = arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA,
        Manifest.permission.WRITE_EXTERNAL_STORAGE)


    private lateinit var webSocketClient: WebSocketClient

    private lateinit var viewFragment: FragmentContainerView

    private lateinit var tvWebsocket: TextView
    private lateinit var tvModel: TextView
    private lateinit var tvPrediction: TextView
    private lateinit var tvVals: TextView

    private lateinit var etIpAddress: EditText

    private lateinit var flowPosition: Flow
    private lateinit var bLaying: Button
    private lateinit var bSitting: Button
    private lateinit var bStanding: Button

    private lateinit var flowControls: Flow
    private lateinit var bCamera: Button
    private lateinit var bRTSP: Button
    private lateinit var bWebcam: Button
    private lateinit var bResetNano: Button

    private var position = -1
    private var reconnectingToWebsocket = false

    private lateinit var mediaPlayer: MediaPlayer

    var recentPrediction = floatArrayOf(0F, 0F, 0F)


    @RequiresApi(Build.VERSION_CODES.R)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE)
        OpenCVLoader.initDebug()

        mediaPlayer = MediaPlayer.create(this, R.raw.ding_sound_effect)

        //force sound to play over speaker
        //this is to correct for latency from bluetooth headphones
        val audioManager = getSystemService(AudioManager::class.java) as AudioManager
        audioManager.mode = AudioManager.MODE_NORMAL
        audioManager.isSpeakerphoneOn = true

        val outputDevices = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
        for (outputDevice in outputDevices) {
            if (outputDevice.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER) {
                //val result = audioManager
                val result = mediaPlayer.setPreferredDevice(outputDevice)
                Log.d("TAG1", result.toString())
                break
            }
        }

        if (!hasPermissions(this, *PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, PERMISSIONS, 1)
        }

        viewFragment = findViewById(R.id.fragment_container_view)

        tvWebsocket = findViewById(R.id.tv_websocket)
        tvModel = findViewById(R.id.tv_model)
        tvPrediction =  findViewById(R.id.tv_prediction)
        tvVals = findViewById(R.id.tv_vals)

        etIpAddress = findViewById(R.id.et_ipaddress)

        //save websocket ip if changed
        etIpAddress.addTextChangedListener(object : TextWatcher {

            override fun afterTextChanged(s: Editable) {}

            override fun beforeTextChanged(s: CharSequence, start: Int,
                                           count: Int, after: Int) {
            }

            override fun onTextChanged(s: CharSequence, start: Int,
                                       before: Int, count: Int) {
                val sharedPref = getPreferences(Context.MODE_PRIVATE) ?: return
                with (sharedPref.edit()) {
                    putString(getString(R.string.websocket_ip_port), s.toString())
                    apply()
                }
            }
        })

        flowPosition = findViewById(R.id.flow_position)
        bLaying = findViewById(R.id.b_laying)
        bLaying.setOnClickListener {
            positionButtonClicked(false, true, true)
            position = 0
        }

        bSitting = findViewById(R.id.b_sitting)
        bSitting.setOnClickListener {
            positionButtonClicked(true, false, true)
            position = 1
        }

        bStanding = findViewById(R.id.b_standing)
        bStanding.setOnClickListener {
            positionButtonClicked(true, true, false)
            position = 2
        }


        //buttons to change fragments
        flowControls = findViewById(R.id.flow_controls)

        bCamera = findViewById(R.id.b_camera)
        bCamera.setOnClickListener {
            var frag = supportFragmentManager.findFragmentByTag("localCamera")
            if (frag == null || !frag.isVisible) {
                supportFragmentManager?.beginTransaction()!!
                    .replace(
                        R.id.fragment_container_view,
                        LocalCameraFragment.newInstance(),
                        "localCamera"
                    )
                    .commit()
            }
        }

        bRTSP = findViewById(R.id.b_rtsp)
        bRTSP.setOnClickListener {
            var frag = supportFragmentManager.findFragmentByTag("rtspCamera")
            if (frag == null || !frag.isVisible) {
                supportFragmentManager?.beginTransaction()!!
                    .replace(
                        R.id.fragment_container_view,
                        RTSPCameraFragment.newInstance(),
                        "rtspCamera"
                    )
                    .commit()
            }
        }

        bWebcam = findViewById(R.id.b_webcam)
        bWebcam.setOnClickListener {
            var wifiManager =
                applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager

            if (wifiManager.isWifiEnabled) {
                var frag = supportFragmentManager.findFragmentByTag("webcam")
                if (frag == null || !frag.isVisible) {
                    supportFragmentManager?.beginTransaction()!!
                        .replace(
                            R.id.fragment_container_view,
                            OutsideStreamFragment.newInstance(),
                            "webcam"
                        )
                        .commit()
                }
            } else {
                //todo: make videoview fragment, oncreate startstream and ondestroy stopstream
            }
        }

        bResetNano = findViewById(R.id.b_restart_nano)
        bResetNano.setOnClickListener {
            sendWebsocketMessage("restart")
        }

        viewFragment.setOnClickListener {
            flowPosition.visibility = if(flowPosition.visibility == Flow.VISIBLE)
                                        Flow.INVISIBLE else Flow.VISIBLE
            flowControls.visibility = if(flowControls.visibility == Flow.VISIBLE)
                                        Flow.INVISIBLE else Flow.VISIBLE
            etIpAddress.visibility  = if(etIpAddress.visibility == EditText.VISIBLE)
                                        EditText.INVISIBLE else EditText.VISIBLE

        }

        //load stored websocket address into edit text
        val sharedPref = getPreferences(Context.MODE_PRIVATE) ?: return
        if (sharedPref.contains(getString(R.string.websocket_ip_port))) {
            etIpAddress.setText(
                sharedPref.getString(
                            getString(R.string.websocket_ip_port),
                            getString(R.string.websocket_ip_port)
                )
            )
        }

        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .add(R.id.fragment_container_view, LocalCameraFragment.newInstance(), "camera")
                .commit()
        }

        connectToWebsocket()
    }

    private fun hasPermissions(context: Context?, vararg permissions: String): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && context != null) {
            for (permission in permissions) {
                if (ActivityCompat.checkSelfPermission(context,
                        permission) != PackageManager.PERMISSION_GRANTED) {
                    return false
                }
            }
        }
        return true
    }

    private fun createWebSocketClient(dogServerUri: URI?) {
        try {
            webSocketClient = object : WebSocketClient(dogServerUri) {

                override fun onOpen(handshakedata: ServerHandshake?) {
                    Log.d("SOCKET", "onOpen")
                    webSocketClient.send("android:1935")
                    runOnUiThread {
                        tvWebsocket.setTextColor(Color.GREEN)
                    }
                }

                override fun onMessage(message: String?) {
                    Log.d("SOCKET", "onMessage: $message")
                    if (message!!.contains("model", true)) {
                        runOnUiThread {
                            if (message!!.contains("true", true)) {
                                tvModel.setTextColor(Color.GREEN)

                                var wifiManager = applicationContext
                                                    .getSystemService(Context.WIFI_SERVICE)
                                                        as WifiManager

                                if (wifiManager.isWifiEnabled) {
                                    var fragmentCamera =
                                        supportFragmentManager.findFragmentByTag("rtspCamera")
                                    if (fragmentCamera != null && fragmentCamera.isVisible) {
                                        (fragmentCamera as RTSPCameraFragment).startStream()
                                    }
                                }
                            } else {
                                tvModel.setTextColor(Color.RED)
                            }
                        }
                    } else if(message!!.contains("prediction", true)) {
                        val split = message!!.split(':')[1].split(',')
                        onPredictionReceived(
                            floatArrayOf(
                                roundPrediction(split[0].toFloat()),
                                roundPrediction(split[1].toFloat()),
                                roundPrediction(split[2].toFloat())
                            )
                        )
                    }
                }

                override fun onClose(code: Int, reason: String?, remote: Boolean) {
                    Log.d("SOCKET", "onClose socket")
                    runOnUiThread {
                        tvWebsocket.setTextColor(Color.RED)
                        tvModel.setTextColor(Color.RED)
                    }
                    var fragmentCamera = supportFragmentManager.findFragmentByTag("camera")
                    if (fragmentCamera != null && fragmentCamera.isVisible) {
                        (fragmentCamera as RTSPCameraFragment).stopStream()
                    }
                    if (!reconnectingToWebsocket) {
                        reconnectingToWebsocket = true
                        connectToWebsocket()
                    }
                }

                override fun onError(ex: Exception?) {
                    Log.e("createWebSocketClient", "onError: ${ex?.message}")
                }

            }
            webSocketClient.connect()
        } catch (e: Exception) {
        }
    }

    private fun positionButtonClicked(laying: Boolean, sitting: Boolean, standing: Boolean) {
        bLaying.isClickable = laying
        bSitting.isClickable = sitting
        bStanding.isClickable = standing

        if(laying) bLaying.setBackgroundColor(Color.LTGRAY)
            else bLaying.setBackgroundColor(Color.DKGRAY)
        if(sitting) bSitting.setBackgroundColor(Color.LTGRAY)
            else bSitting.setBackgroundColor(Color.DKGRAY)
        if(standing) bStanding.setBackgroundColor(Color.LTGRAY)
            else bStanding.setBackgroundColor(Color.DKGRAY)
    }

    fun sendWebsocketMessage(message : String) {
        if (webSocketClient.isOpen)
            webSocketClient.send("android: $message")
    }

    private fun connectToWebsocket() {
        val dogServerURI = getWebsocketURI()
        Thread {
            createWebSocketClient(dogServerURI)
            while (!webSocketClient.isOpen) {
                createWebSocketClient(dogServerURI)
                Thread.sleep(2000)
            }
            reconnectingToWebsocket = false
        }.start()
    }

    fun getWebsocketURI(): URI {
        val websocketIpPort: String
        val sharedPref = getPreferences(Context.MODE_PRIVATE)
        if (sharedPref.contains(getString(R.string.websocket_ip_port))) {
            websocketIpPort = sharedPref.getString(
                getString(R.string.websocket_ip_port),
                getString(R.string.websocket_ip_port)
            )!!
        } else {
            websocketIpPort = getString(R.string.websocket_ip_port)
        }
        return URI("ws://$websocketIpPort/ws/")
    }

    //assuming a float array with 3 indices
    //laying, sitting, standing
    fun onPredictionReceived(prediction: FloatArray) {
        tvVals.text = "${"%.2f,%.2f,%.2f".format(prediction[0], prediction[1], prediction[2])}"
        if (position < 0)
            return

        //play sound if was adhering to position, but no longer is
        if (recentPrediction[position] > PREDICTION_CUTOFF &&
            prediction[position] <= PREDICTION_CUTOFF) {
            mediaPlayer.start()
        }

        runOnUiThread {
            if (prediction[position] > PREDICTION_CUTOFF) {
                tvPrediction.setTextColor(Color.GREEN)
                tvVals.setTextColor(Color.GREEN)
            } else {
                tvPrediction.setTextColor(Color.RED)
                tvVals.setTextColor(Color.RED)
            }

        }

        recentPrediction = prediction.copyOf()
    }

    companion object {
        fun roundPrediction(value: Float): Float {
            val df = DecimalFormat("#.##")
            df.roundingMode = RoundingMode.HALF_UP
            return df.format(value).toFloat()
        }
    }

    override fun onDestroy() {
        sendWebsocketMessage("close")
        Thread.sleep(1000)
        webSocketClient.close()
        mediaPlayer.release()
        super.onDestroy()
    }
}