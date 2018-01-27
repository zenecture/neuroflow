package neuroflow.application.processor

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.net.URL
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.common.Logs
import neuroflow.core.Network.{Matrix, Vector}

import scala.io.Source

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Image extends Logs {


  /**
    * Loads image from `file` or `path` and returns flattened sequence
    * of all color channels and pixels, where values are normalized to be <= 1.0.
    */

  def extractRgb(path: String): Vector[Double] = extractRgb(new File(path))

  def extractRgb(file: File): Vector[Double] = {
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
    * Loads image from `url`, `path` or `file` and returns a matrix,
    * where image is linearized using column major into a full row per color channel.
    * rgb values are scaled from [0, 255] to [0.0, 1.0].
    */

  def extractRgb3d(url: URL): Matrix[Double] = extractRgb3d(ImageIO.read(url))

  def extractRgb3d(path: String): Matrix[Double] = extractRgb3d(new File(path))

  def extractRgb3d(file: File): Matrix[Double] = extractRgb3d(ImageIO.read(file))

  def extractRgb3d(img: BufferedImage): Matrix[Double] = {
    val (w, h) = (img.getWidth, img.getHeight)
    val out = DenseMatrix.zeros[Double](3, w * h)
    (0 until h).foreach { _h =>
      (0 until w).foreach { _w =>
        val c = new Color(img.getRGB(_w, _h))
        val r = c.getRed   / 255.0
        val g = c.getGreen / 255.0
        val b = c.getBlue  / 255.0
        out.update(0, _w * h + _h, r)
        out.update(1, _w * h + _h, g)
        out.update(2, _w * h + _h, b)
      }
    }
    out
  }


  /**
    * Loads portable gray map as flat vector
    */

  def extractPgm(path: String): Vector[Double] = extractPgm(new File(path))

  def extractPgm(file: File): Vector[Double] = {
    val raw = Source.fromFile(file).getLines.drop(2).toArray // P2, width, height
    val max = raw.head.toDouble
    val img = raw.tail.flatMap(_.split(" ")).map(_.toDouble / max)
    DenseVector(img)
  }


  /**
    * Loads image from `file` or `path` and returns flattened sequence of pixels,
    * activated based on `selector` result
    */

  def extractBinary(path: String, selector: Int => Boolean): Vector[Double] = extractBinary(new File(path), selector)

  def extractBinary(file: File, selector: Int => Boolean): Vector[Double] = {
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
