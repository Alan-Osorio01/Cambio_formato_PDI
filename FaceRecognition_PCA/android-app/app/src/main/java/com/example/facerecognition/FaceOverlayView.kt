package com.example.facerecognition

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class FaceOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val paint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.MAGENTA
    }

    private var rect: RectF? = null

    fun setRect(r: RectF, recognized: Boolean) {
        rect = r
        paint.color = if (recognized) Color.GREEN else Color.RED
        invalidate()
    }

    fun clear() {
        rect = null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        rect?.let { canvas.drawRect(it, paint) }
    }
}
