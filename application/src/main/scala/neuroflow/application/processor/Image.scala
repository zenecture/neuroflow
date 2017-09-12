package neuroflow.application.processor

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.net.URL
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.common.Logs
import neuroflow.core.Network._

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
    * Loads image from `url`, `path` or `file` and returns a 3d rgb-volume,
    * where color channel values are normalized to be <= 1.0.
    * Optionally, a `dimension` can be enforced through zero-padding and cut-off.
    */

  def extractRgb3d(url: URL, dimension: Option[(Int, Int)]): Matrices =
    extractRgb3d(ImageIO.read(url), dimension)

  def extractRgb3d(path: String, dimension: Option[(Int, Int)]): Matrices =
    extractRgb3d(new File(path), dimension)

  def extractRgb3d(file: File, dimension: Option[(Int, Int)]): Matrices =
    extractRgb3d(ImageIO.read(file), dimension)

  def extractRgb3d(img: BufferedImage, dimension: Option[(Int, Int)]): Matrices = {
    val (w, h) = (img.getWidth, img.getHeight)
    val out = Array.fill(3)(DenseMatrix.zeros[Double](h, w))
    (0 until h).foreach { _h =>
      (0 until w).foreach { _w =>
        val c = new Color(img.getRGB(_w, _h))
        val r = c.getRed   / 255.0
        val g = c.getGreen / 255.0
        val b = c.getBlue  / 255.0
        out(0).update(_h, _w, r)
        out(1).update(_h, _w, g)
        out(2).update(_h, _w, b)
      }
    }
    dimension match {

      case Some((x, y)) if x == w && y == w => out

      case Some((x, y)) if x == w && y  > h =>
        val (_x, _y) = (x.toDouble, y.toDouble)
        val dY1 = math.floor((_y - h) / 2.0).toInt
        val dY2 =  math.ceil((_y - h) / 2.0).toInt
        val mY1 = DenseMatrix.zeros[Double](dY1, x)
        val mY2 = DenseMatrix.zeros[Double](dY2, x)

        out.map { m =>
          DenseMatrix.vertcat(mY1, m, mY2)
        }

      case Some((x, y)) if x  > w && y == h =>
        val (_x, _y) = (x.toDouble, y.toDouble)
        val dX1 = math.floor((_x - w) / 2.0).toInt
        val dX2 =  math.ceil((_x - w) / 2.0).toInt
        val mX1 = DenseMatrix.zeros[Double](y, dX1)
        val mX2 = DenseMatrix.zeros[Double](y, dX2)

        out.map { m =>
          DenseMatrix.horzcat(mX1, m, mX2)
        }

      case Some((x, y)) if x  > w && y  < h =>
        val (_x, _y) = (x.toDouble, y.toDouble)
        val dX1 = math.floor((_x - w) / 2.0).toInt
        val dX2 =  math.ceil((_x - w) / 2.0).toInt
        val mX1 = DenseMatrix.zeros[Double](y, dX1)
        val mX2 = DenseMatrix.zeros[Double](y, dX2)

        out.map { m =>
          DenseMatrix.horzcat(mX1, m(0 until (h - 1), 0 until (w - 1)), mX2)
        }

      case Some((x, y)) if x  > w && y  > h =>
        val (_x, _y) = (x.toDouble, y.toDouble)
        val dX1 = math.floor((_x - w) / 2.0).toInt
        val dX2 =  math.ceil((_x - w) / 2.0).toInt
        val dY1 = math.floor((_y - h) / 2.0).toInt
        val dY2 =  math.ceil((_y - h) / 2.0).toInt

        val mX1 = DenseMatrix.zeros[Double](h, dX1)
        val mX2 = DenseMatrix.zeros[Double](h, dX2)
        val mY1 = DenseMatrix.zeros[Double](dY1, dX1 + w + dX2)
        val mY2 = DenseMatrix.zeros[Double](dY2, dX1 + w + dX2)

        out.map { m =>
          DenseMatrix.vertcat(mY1, DenseMatrix.horzcat(mX1, m, mX2), mY2)
        }

      case Some((x, y)) if x <= w && y <= h =>
        out.map { m => m(0 until (y - 1), 0 until (x - 1)) }

      case Some((x, y))                     =>
        info (s"Can't resize image. (x, y, h, w) = ($x, $y, $h, $w)")
        out

      case None                             => out

    }
  }


  /**
    * Loads portable gray map as flat vector
    */

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
