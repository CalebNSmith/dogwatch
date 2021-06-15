package com.caleb.dogwatch

import android.os.Bundle
import android.util.Log
import android.view.*
import androidx.fragment.app.Fragment
import com.pedro.rtsp.utils.ConnectCheckerRtsp
import com.pedro.rtspserver.RtspServerCamera1

/**
 * This fragment creates an rtsp stream of the local camera that will be recieved by a server to run
 * inference on. This will not work on mobile networks due to not having permissions to open ports.
 *
 * Use LocalCameraFragment to achieve lowest latency for inference
 */
class RTSPCameraFragment : Fragment(R.layout.fragment_camera_stream),
                            ConnectCheckerRtsp, SurfaceHolder.Callback {

    private lateinit var surfaceView : SurfaceView
    private var rtspServerCamera1: RtspServerCamera1? = null

    companion object {
        fun newInstance(): RTSPCameraFragment {
            return RTSPCameraFragment()
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        surfaceView = view?.findViewById(R.id.surface_view) as SurfaceView
        rtspServerCamera1 = RtspServerCamera1(surfaceView, this, 1935)
        startStream()
        surfaceView.holder.addCallback(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        rtspServerCamera1!!.stopStream()
        rtspServerCamera1!!.stopPreview()
    }

    override fun onAuthErrorRtsp() {
    }

    override fun onAuthSuccessRtsp() {
    }

    override fun onConnectionFailedRtsp(reason: String) {
    }

    override fun onConnectionSuccessRtsp() {
    }

    override fun onDisconnectRtsp() {
    }

    override fun onNewBitrateRtsp(bitrate: Long) {
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        rtspServerCamera1!!.startPreview()
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
    }

    fun startStream() {
        if (!rtspServerCamera1!!.isStreaming &&
                rtspServerCamera1!!.prepareAudio() &&
                rtspServerCamera1!!.prepareVideo()) {
            rtspServerCamera1!!.startStream()
            var homeActivity = activity
            if (homeActivity is MainActivity) {
                homeActivity.sendWebsocketMessage(rtspServerCamera1!!.getEndPointConnection())
            }
        }
    }

    fun stopStream() {
        if (!rtspServerCamera1!!.isStreaming) {
            rtspServerCamera1!!.stopStream()
        }
    }

}