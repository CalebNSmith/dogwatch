package com.caleb.dogwatch

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.fragment.app.Fragment
import org.opencv.android.Utils
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import java.lang.Exception
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.SocketException
import java.net.URI

/**
 * This fragment is to display a stream from a source outside of the device. It receives frames via
 * a socket, which will not work on mobile networks due to not having permission to open ports.
 */
class OutsideStreamFragment : Fragment(R.layout.fragment_outside_stream) {

    private lateinit var dataGramSocket: DatagramSocket
    private lateinit var iv_stream: ImageView

    companion object {
        fun newInstance(): OutsideStreamFragment {
            return OutsideStreamFragment()
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        var homeActivity = activity
        if (homeActivity is MainActivity) {
            homeActivity.sendWebsocketMessage("start stream")
        }

        iv_stream = view?.findViewById(R.id.iv_stream) as ImageView
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread {
            try {
                dataGramSocket = DatagramSocket(12345)
                var imageBuffer = ByteArray(200000)
                var lastIndex = 0
                var totalImageBytes = 0
                while (true) {
                    var packetBuffer = ByteArray(65536) //2^16 max packet size
                    var packet = DatagramPacket(packetBuffer, packetBuffer.size)

                    try {
                        dataGramSocket.receive(packet)
                        //get number of packets remaining to compose the image
                        var numPackets = packet.data[0].toInt()

                        if (numPackets > 1) {
                            //copy packet data into image byte buffer
                            packetBuffer.copyInto(
                                                    imageBuffer,
                                                    lastIndex,
                                                    1,
                                                    packet.length + 1
                                                )
                            lastIndex = packet.length + 1
                            totalImageBytes += packet.length - 1
                        } else {
                            //copy packet data into image byte buffer
                            packetBuffer.copyInto(
                                                    imageBuffer,
                                                    lastIndex,
                                                    1,
                                                    packet.length + 1
                                                )
                            totalImageBytes += packet.length - 1

                            //convert image byte buffer to bitmap
                            //and set image view
                            var matOfByte = MatOfByte()
                            matOfByte.fromList(imageBuffer.toList().subList(0, totalImageBytes + 1))
                            var img = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_COLOR)
                            if (img.cols() > 0 && img.rows() > 0) {
                                var bitmap = Bitmap.createBitmap(
                                    img.cols(),
                                    img.rows(),
                                    Bitmap.Config.ARGB_8888
                                )
                                Utils.matToBitmap(img, bitmap)
                                requireActivity().runOnUiThread {
                                    iv_stream.setImageBitmap(bitmap)
                                }
                            }
                            imageBuffer = ByteArray(200000)
                            lastIndex = 0
                            totalImageBytes = 0
                        }
                    } catch(e: Exception) {
                        imageBuffer = ByteArray(200000)
                        lastIndex = 0
                        totalImageBytes = 0
                    }
                }
                dataGramSocket.close()
            } catch (e: Exception) {
            }
        }.start()
    }

    fun setImage(bitmap: Bitmap) {
        requireActivity().runOnUiThread {
            iv_stream.setImageBitmap(bitmap)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        var homeActivity = activity
        if (homeActivity is MainActivity) {
            homeActivity.sendWebsocketMessage("end stream")
        }
    }
}