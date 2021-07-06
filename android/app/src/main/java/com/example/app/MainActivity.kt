package com.example.app

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.app.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //
        val model = Model.newInstance(applicationContext)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1), DataType.FLOAT32)
        //val byteBuffer = ByteBuffer.allocate(4).putFloat(67000.0f)
        //var byteBuffer: ByteBuffer = ByteBuffer.wrap(FloatArray2ByteArray(floatArrayOf(16000.0f)))

        //inputFeature0.loadBuffer(byteBuffer)
        inputFeature0.loadArray(floatArrayOf(16000.0F))

        inputFeature0.floatArray.forEach {
            //Log.d("result", it.toString())
        }


        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        model.close()

        //Log.d("result", "HOLA")
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer


        Toast.makeText(this,"output: "+outputFeature0.floatArray[0], Toast.LENGTH_SHORT).show();
        outputFeature0.floatArray.forEach {
            Log.d("result__", it.toString())
        }


        // Releases model resources if no longer used.


        //


    }
}