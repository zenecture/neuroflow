package neuroflow.application.processor

import java.awt.Color
import java.io.File
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.core.Network._

import scala.io.Source

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Image {

  /**
    * Loads image from `file` or `path` and returns flattened sequence
    * of all color channels and pixels, where values are normalized to be <= 1.0.
    */
  def extractRgb(path: String): Vector = extractRgb(new File(path))
  def extractRgb(file: File): Vector = {
    val img = ImageIO.read(file)
    val res =
      (0 until img.getHeight).flatMap { h =>
        (0 until img.getWidth).flatMap { w =>
          val c = new Color(img.getRGB(w, h))
          c.getRed / 255.0 :: c.getGreen / 255.0 :: c.getBlue / 255.0 :: Nil
        }
      }
    DenseVector(res.toArray)
  }

  /**
    * Loads image from `file` or `path` and returns a 3d rgb-volume,
    * where values are normalized to be <= 1.0.
    */
  def extractRgb3d(path: String): Matrices = extractRgb3d(new File(path))
  def extractRgb3d(file: File): Matrices = {
    val img = ImageIO.read(file)
    val (w, h) = (img.getWidth, img.getHeight)
    val out = Array.fill(3)(DenseMatrix.zeros[Double](w, h))
    (0 until h).foreach { _h =>
      (0 until w).foreach { _w =>
        val c = new Color(img.getRGB(_w, _h))
        val r = c.getRed    / 255.0
        val g = c.getGreen  / 255.0
        val b = c.getBlue   / 255.0
        out(0).update(_w, _h, r)
        out(1).update(_w, _h, g)
        out(2).update(_w, _h, b)
      }
    }
    out
  }

  def extractPgm(path: String): Vector = extractPgm(new File(path))
  def extractPgm(file: File): Vector = {
    val raw = Source.fromFile(file).getLines.drop(2).toArray // P2, width, height
    val max = raw.head.toDouble
    val img = raw.tail.flatMap(_.split(" ")).map(_.toDouble / max)
    DenseVector(img)
  }

  /**
    * Loads image from `file` or `path` and returns flattened sequence of pixels,
    * activated based on `selector` result
    */
  def extractBinary(path: String, selector: Int => Boolean): Vector = extractBinary(new File(path), selector)
  def extractBinary(file: File, selector: Int => Boolean): Vector = {
    val img = ImageIO.read(file)
    val res =
      (0 until img.getHeight).flatMap { h =>
        (0 until img.getWidth).flatMap { w =>
          val c = new Color(img.getRGB(w, h))
          (if (selector(c.getRed) || selector(c.getBlue) || selector(c.getGreen)) 1.0 else 0.0) :: Nil
        }
      }
    DenseVector(res.toArray)
  }

}
