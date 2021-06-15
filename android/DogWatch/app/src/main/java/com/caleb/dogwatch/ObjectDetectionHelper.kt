/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.example.camerax.tflite

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer
import java.nio.ByteOrder


/**
 * Helper class used to communicate between our app and the TF object detection model
 */
class ObjectDetectionHelper(private val tflite: Interpreter, private val labels: List<String>) {

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    data class ObjectPrediction(val score: Float)

    private val labelIndices =  arrayOf(FloatArray(OBJECT_COUNT))
    private val scores =  arrayOf(FloatArray(OBJECT_COUNT))

    private val outputBuffer = mapOf(
        0 to scores
    )


    fun predict(image: TensorImage): List<Float> {
        tflite.runForMultipleInputsOutputs(arrayOf(image.buffer), outputBuffer)
        return listOf(scores[0][0], scores[0][1], scores[0][2])
    }

    companion object {
        const val OBJECT_COUNT = 3
    }
}