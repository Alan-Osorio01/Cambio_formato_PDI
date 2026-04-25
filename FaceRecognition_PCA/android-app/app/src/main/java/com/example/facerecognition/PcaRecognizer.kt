package com.example.facerecognition

import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Reconocimiento multi-clase por distancia en eigenspace (Eigenfaces clásico).
 *
 * Entrenamiento: Alan (290 imágenes aumentadas) + Other (105 caras del dataset).
 * Reconocimiento: proyectar cara → encontrar vecino más cercano → si es "Alan" → acceso.
 *
 * Solo se usan los top MAX_EIGEN eigenvectores (los más discriminativos) para
 * reducir tiempo de carga y memoria sin perder precisión.
 */
class PcaRecognizer(context: Context) {

    companion object {
        private const val AUTHORIZED_ID = "Alan"
        private const val MAX_EIGEN = 80          // top eigenvectores a usar
        private const val THRESHOLD = 500_000.0   // respaldo: cara que no es ninguna persona conocida
    }

    private val numFaces: Int
    private val avgVec: Mat          // 10000 × 1
    private val eigenVec: Mat        // MAX_EIGEN × 10000
    private val faceIds: List<String>
    private val facesInEigen: Mat    // MAX_EIGEN × numFaces

    init {
        val ids = mutableListOf<String>()
        val rawCols = mutableListOf<FloatArray>()

        context.assets.open("facesdata.txt").bufferedReader().forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val colon = line.indexOf(':')
            ids.add(line.substring(0, colon))
            val vals = line.substring(colon + 1).trim().split(" ").filter { it.isNotBlank() }
            rawCols.add(FloatArray(minOf(vals.size, MAX_EIGEN)) { vals[it].toFloat() })
        }

        numFaces = ids.size
        faceIds = ids

        // Proyecciones de entrenamiento: solo las primeras MAX_EIGEN dimensiones
        facesInEigen = Mat.zeros(MAX_EIGEN, numFaces, CvType.CV_32FC1)
        for (col in 0 until numFaces) {
            val v = rawCols[col]
            for (row in 0 until v.size)
                facesInEigen.put(row, col, v[row].toDouble())
        }

        avgVec = Mat.zeros(10000, 1, CvType.CV_32FC1)
        context.assets.open("mean.txt").bufferedReader().readLine()
            ?.trim()?.split(" ")?.filter { it.isNotBlank() }
            ?.forEachIndexed { i, v -> if (i < 10000) avgVec.put(i, 0, v.toDouble()) }

        // Leer solo los primeros MAX_EIGEN eigenvectores del archivo
        eigenVec = Mat.zeros(MAX_EIGEN, 10000, CvType.CV_32FC1)
        var row = 0
        context.assets.open("eigen.txt").bufferedReader().forEachLine { line ->
            if (line.isBlank() || row >= MAX_EIGEN) return@forEachLine
            line.trim().split(" ").filter { it.isNotBlank() }.forEachIndexed { col, v ->
                if (col < 10000) eigenVec.put(row, col, v.toDouble())
            }
            row++
        }
    }

    fun recognize(grayFace: Mat): String {
        // 1. Preprocesar
        val resized = Mat()
        Imgproc.resize(grayFace, resized, Size(100.0, 100.0))
        Imgproc.equalizeHist(resized, resized)
        resized.convertTo(resized, CvType.CV_32FC1)
        val faceVec = resized.reshape(1, 10000)  // 10000 × 1

        // 2. Proyectar al eigenspace (MAX_EIGEN dimensiones)
        val diff = Mat()
        Core.subtract(faceVec, avgVec, diff)
        val projected = Mat()
        Core.gemm(eigenVec, diff, 1.0, Mat(), 0.0, projected)  // MAX_EIGEN × 1

        // 3. Vecino más cercano entre las proyecciones de entrenamiento
        var bestId = faceIds[0]
        var bestDist = Double.MAX_VALUE
        for (col in 0 until numFaces) {
            val dist = Core.norm(facesInEigen.col(col), projected, Core.NORM_L2)
            if (dist < bestDist) { bestDist = dist; bestId = faceIds[col] }
        }

        // 4. Acceso solo si el vecino más cercano es la persona autorizada
        if (bestDist > THRESHOLD) return "None"
        return if (bestId == AUTHORIZED_ID) AUTHORIZED_ID else "None"
    }
}
