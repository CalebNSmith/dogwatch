package com.caleb.dogwatch

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.TextView
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.*
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.LifecycleOwner
import com.android.example.camerax.tflite.ObjectDetectionHelper
import com.android.example.camerax.tflite.YuvToRgbConverter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.math.RoundingMode
import java.text.DecimalFormat
import java.util.concurrent.Executors

/**
 * This fragment loads /assets/model.tflite and captures the image from the back facing camera
 * to perform inference on the image
 *
 * A preview of what the camera sees is displayed
 */
class LocalCameraFragment: Fragment(R.layout.fragment_camera_local) {

    private lateinit var container: ConstraintLayout
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var viewFinder: PreviewView

    private val executor = Executors.newSingleThreadExecutor()

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK

    private var imageRotationDegrees: Int = 0
    private val tfImageBuffer = TensorImage(DataType.FLOAT32)

    private var isPaused = false

    private val tfImageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize)) //center crop
            .add(
                ResizeOp(
                tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR)
            )
            .add(Rot90Op(imageRotationDegrees / 90))
            .add(NormalizeOp(0f, 1f))
            .build()
    }

    private val tflite by lazy {
        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply{
            if(compatList.isDelegateSupportedOnThisDevice){
                // if the device has a supported GPU, add the GPU delegate
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                // if the GPU is not supported, run on 4 threads
                this.setNumThreads(4)
            }
            setUseNNAPI(false)
        }

        Interpreter(
            FileUtil.loadMappedFile(activity?.applicationContext!!, MODEL_PATH),
            Interpreter.Options().setUseNNAPI(false))
    }

    private val detector by lazy {
        ObjectDetectionHelper(
                                tflite,
                                FileUtil.loadLabels(activity?.applicationContext!!, LABELS_PATH)
                            )
    }

    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        container = view?.findViewById(R.id.camera_container)
        viewFinder = view?.findViewById(R.id.view_finder)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        isPaused = false
    }


    @SuppressLint("UnsafeExperimentalUsageError")
    private fun bindCameraUseCases() = viewFinder.post {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(activity?.applicationContext!!)
        cameraProviderFuture.addListener(Runnable {

            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()

            // Set up the view finder use case to display camera preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(viewFinder.display.rotation)
                .build()

            // Set up the image analysis use case which will process frames in real time
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            var frameCounter = 0
            var lastFpsTimestamp = System.currentTimeMillis()
            val converter = YuvToRgbConverter(activity?.applicationContext!!)

            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { image ->
                if (!::bitmapBuffer.isInitialized) {
                    // The image rotation and RGB image buffer are initialized only once
                    // the analyzer has started running
                    imageRotationDegrees = image.imageInfo.rotationDegrees
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width, image.height, Bitmap.Config.ARGB_8888)
                }

                if (!isPaused) {


                    // Convert the image to RGB and place it in our shared buffer
                    image.use { converter.yuvToRgb(image.image!!, bitmapBuffer) }

                    // Process the image in Tensorflow
                    val tfImage =
                        tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })

                    // Perform the object detection for the current frame
                    val predictions = detector.predict(tfImage)


                    // Report only the top prediction
                    reportPrediction(predictions)
                }
            })

            // Create a new camera selector each time, enforcing lens facing
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            // Apply declared configs to CameraX using the same lifecycle owner
            cameraProvider.unbindAll()
            val camera = cameraProvider.bindToLifecycle(
                this as LifecycleOwner, cameraSelector, preview, imageAnalysis)

            // Use the camera object to link our preview use case with the view
            preview.setSurfaceProvider(viewFinder.createSurfaceProvider(camera.cameraInfo))

        }, ContextCompat.getMainExecutor(activity?.applicationContext!!))
    }

    private fun reportPrediction(predictions: List<Float>) = viewFinder.post {
        if (activity is MainActivity)
            (activity as MainActivity)
                .onPredictionReceived(
                    floatArrayOf(
                            MainActivity.roundPrediction(predictions[0]),
                            MainActivity.roundPrediction(predictions[1]),
                            MainActivity.roundPrediction(predictions[2])
                    )
                )
    }


    override fun onPause() {
        super.onPause()
        isPaused = true
    }

    override fun onResume() {
        super.onResume()
        bindCameraUseCases()
        isPaused = false
    }


    companion object {
        private const val MODEL_PATH = "model.tflite"
        private const val LABELS_PATH = "labels.txt"

        fun newInstance(): LocalCameraFragment {
            return LocalCameraFragment()
        }
    }
}