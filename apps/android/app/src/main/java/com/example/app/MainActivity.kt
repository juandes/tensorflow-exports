package com.example.app

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.app.ml.Model
import com.example.app.ml.ModelFloat16Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    //val model: Model = Model.newInstance(applicationContext)
    lateinit var model: Model
    lateinit var quantModel: ModelFloat16Quant


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        model = Model.newInstance(applicationContext)
        quantModel = ModelFloat16Quant.newInstance(applicationContext)

        //
        //val model =

        // Creates inputs for reference.
        //val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1), DataType.FLOAT32)
        //inputFeature0.loadArray(floatArrayOf(16000.0F))



        // Runs model inference and gets result.
        //val outputs = model.process(inputFeature0)
        //model.close()
        //val outputFeature0 = outputs.outputFeature0AsTensorBuffer


        //Toast.makeText(this,"output: "+outputFeature0.floatArray[0], Toast.LENGTH_SHORT).show();
        //outputFeature0.floatArray.forEach {
          //  Log.d("result__", it.toString())
        //}


        // Releases model resources if no longer used.


        //


    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    fun predict(view: View) {
        val editInput = findViewById<EditText>(R.id.editInput)
        val input = editInput.text.toString().toFloat()

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1), DataType.FLOAT32)
        inputFeature0.loadArray(floatArrayOf(input))

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)

        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        findViewById<TextView>(R.id.outputView).text = outputFeature0.floatArray[0].toString()
        outputFeature0.floatArray.forEach {
          Log.d("result__", it.toString())
        }

        //
        //val editInput = findViewById<EditText>(R.id.editInput)
        //val input = editInput.text.toString().toFloat()

        //val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1), DataType.FLOAT32)
        //inputFeature0.loadArray(floatArrayOf(input))

        // Runs model inference and gets result.
        val quantOutputs = quantModel.process(inputFeature0)

        val quantOutputFeature0 = quantOutputs.outputFeature0AsTensorBuffer
        findViewById<TextView>(R.id.quantOutputView).text = quantOutputFeature0.floatArray[0].toString()
        quantOutputFeature0.floatArray.forEach {
            Log.d("result__", it.toString())
        }
    }
}