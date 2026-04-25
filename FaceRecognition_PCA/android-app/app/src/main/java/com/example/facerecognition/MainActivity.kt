package com.example.facerecognition

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.facerecognition.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var faceDetector: CascadeClassifier
    private lateinit var recognizer: PcaRecognizer

    private var mode = Mode.RECOGNISE
    private val capturedFaces = mutableListOf<Mat>()
    // Ventana de 20 frames para suavizar el resultado — evita parpadeo
    private val recentIds = ArrayDeque<String>(20)
    private var lastShownId = ""
    private var noFaceCount = 0

    enum class Mode { RECOGNISE, CAPTURE }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV no cargó", Toast.LENGTH_LONG).show()
            return
        }

        faceDetector = loadCascade("haarcascade_frontalface_default.xml")
        recognizer = PcaRecognizer(this)

        if (hasCameraPermission()) startCamera()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)

        binding.btnRecognize.setOnClickListener {
            mode = Mode.RECOGNISE
            binding.tvStatus.text = "Modo: Reconocer"
        }
        binding.btnCapture.setOnClickListener {
            if (mode != Mode.CAPTURE) {
                mode = Mode.CAPTURE
                capturedFaces.clear()
                binding.tvStatus.text = "Modo: Captura — pon tu cara y presiona Capture"
            } else {
                binding.tvStatus.text = "Capturadas: ${capturedFaces.size} cara(s)"
            }
        }
        binding.btnTrain.setOnClickListener {
            if (capturedFaces.isEmpty()) {
                Toast.makeText(this, "Primero captura caras", Toast.LENGTH_SHORT).show()
            } else {
                askForIdAndSave()
            }
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = binding.previewView.surfaceProvider
            }
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build().also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

            provider.unbindAll()
            try {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analyzer)
            } catch (e: Exception) {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analyzer)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val rotation = imageProxy.imageInfo.rotationDegrees
        val rawMat = imageProxy.toGrayMat()
        imageProxy.close()

        // Rotar el frame para que la cara quede vertical (el sensor del celular es landscape)
        val upright = rotateMat(rawMat, rotation)

        // Ecualización de histograma → mejora detección con poca luz o alto contraste
        Imgproc.equalizeHist(upright, upright)

        val faces = MatOfRect()
        // minNeighbors=4 (más tolerante), minSize=100px (evita falsos positivos pequeños)
        faceDetector.detectMultiScale(upright, faces, 1.1, 4, 0,
            Size(100.0, 100.0), Size(600.0, 600.0))
        val rects = faces.toArray()

        if (rects.isNotEmpty()) {
            noFaceCount = 0
            val r = rects.maxByOrNull { it.width * it.height }!!
            val faceRoi = Mat(upright, r)

            when (mode) {
                Mode.RECOGNISE -> {
                    val id = recognizer.recognize(faceRoi)
                    val smoothId = smoothId(id)
                    val scaledRect = scaleRect(r, upright.cols(), upright.rows())
                    // Solo actualizar UI si el resultado cambia (evita parpadeo)
                    if (smoothId != lastShownId) {
                        lastShownId = smoothId
                        runOnUiThread { showResult(smoothId, scaledRect) }
                    } else {
                        runOnUiThread { binding.overlayView.setRect(scaledRect, smoothId != "None") }
                    }
                }
                Mode.CAPTURE -> {
                    val saved = Mat(); faceRoi.copyTo(saved)
                    capturedFaces.add(saved)
                    mode = Mode.RECOGNISE
                    runOnUiThread {
                        binding.tvStatus.text = "Capturadas: ${capturedFaces.size} cara(s). Sigue o presiona Train."
                    }
                }
            }
        } else {
            noFaceCount++
            // Esperar 30 frames sin cara antes de limpiar pantalla (evita parpadeo momentáneo)
            if (noFaceCount > 30) {
                noFaceCount = 0
                lastShownId = ""
                recentIds.clear()
                runOnUiThread {
                    binding.overlayView.clear()
                    binding.tvRecognizedId.text = "Buscando cara..."
                    binding.tvRecognizedId.setBackgroundColor(Color.parseColor("#AA000000"))
                }
            }
        }
    }

    private fun showResult(id: String, rect: android.graphics.RectF) {
        if (id == "None") {
            binding.tvRecognizedId.text = "✗  ACCESO DENEGADO"
            binding.tvRecognizedId.setBackgroundColor(Color.parseColor("#CC990000"))
            binding.overlayView.setRect(rect, recognized = false)
        } else {
            binding.tvRecognizedId.text = "✓  ACCESO CONCEDIDO\n    $id"
            binding.tvRecognizedId.setBackgroundColor(Color.parseColor("#CC006600"))
            binding.overlayView.setRect(rect, recognized = true)
        }
    }

    private fun rotateMat(mat: Mat, degrees: Int): Mat {
        if (degrees == 0) return mat
        val rotated = Mat()
        val code = when (degrees) {
            90  -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            270 -> Core.ROTATE_90_COUNTERCLOCKWISE
            else -> return mat
        }
        Core.rotate(mat, rotated, code)
        return rotated
    }

    private fun scaleRect(r: Rect, matW: Int, matH: Int): android.graphics.RectF {
        val sx = binding.previewView.width.toFloat() / matW
        val sy = binding.previewView.height.toFloat() / matH
        return android.graphics.RectF(r.x * sx, r.y * sy,
            (r.x + r.width) * sx, (r.y + r.height) * sy)
    }

    private fun smoothId(id: String): String {
        recentIds.addLast(id)
        if (recentIds.size > 20) recentIds.removeFirst()
        // Necesita mayoría de más del 60% para cambiar resultado
        val counts = recentIds.groupingBy { it }.eachCount()
        val best = counts.maxByOrNull { it.value } ?: return id
        return if (best.value >= recentIds.size * 0.6) best.key else lastShownId.ifEmpty { id }
    }

    private fun askForIdAndSave() {
        val input = android.widget.EditText(this).apply { hint = "Nombre (ej: Alan)" }
        AlertDialog.Builder(this)
            .setTitle("Guardar cara")
            .setMessage("${capturedFaces.size} cara(s). Ingresa un nombre:")
            .setView(input)
            .setPositiveButton("Guardar") { _, _ ->
                val id = input.text.toString().trim()
                if (id.isBlank()) return@setPositiveButton
                val dir = File(filesDir, "faces").also { it.mkdirs() }
                capturedFaces.forEachIndexed { i, mat ->
                    org.opencv.imgcodecs.Imgcodecs.imwrite(File(dir, "${id}_$i.bmp").absolutePath, mat)
                }
                binding.tvStatus.text = "Guardadas ${capturedFaces.size} cara(s) para '$id'. Re-entrena en PC."
                capturedFaces.clear()
                mode = Mode.RECOGNISE
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

    private fun loadCascade(name: String): CascadeClassifier {
        val file = File(cacheDir, name)
        assets.open(name).use { i -> FileOutputStream(file).use { i.copyTo(it) } }
        return CascadeClassifier(file.absolutePath)
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(code: Int, perms: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == 100 && results.isNotEmpty() && results[0] == PackageManager.PERMISSION_GRANTED)
            startCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

fun ImageProxy.toGrayMat(): Mat {
    val buf = planes[0].buffer
    val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
    return Mat(height, width, CvType.CV_8UC1).also { it.put(0, 0, bytes) }
}
