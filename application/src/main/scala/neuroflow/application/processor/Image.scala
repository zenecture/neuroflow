package neuroflow.application.processor

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.net.URL
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero
import neuroflow.common.{Logs, Tensor, Tensorish}

import scala.io.Source
import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Image extends Logs {


  /**
    * Loads image from `file` or `path` and returns flattened sequence
    * of all color channels and pixels, where values are normalized to be <= 1.0.
    */

  def extractRgb(path: String): DenseVector[Double] = extractRgb(new File(path))

  def extractRgb(file: File): DenseVector[Double] = {
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
    * Loads image from `url`, `path` or `file` and returns a [[RgbTensor]].
    * All pixel colors are scaled from [0, 255] to [0.0, 1.0].
    */

  def extractRgb3d(url: URL): Tensor[Double] = extractRgb3d(ImageIO.read(url))

  def extractRgb3d(path: String): Tensor[Double] = extractRgb3d(new File(path))

  def extractRgb3d(file: File): Tensor[Double] = extractRgb3d(ImageIO.read(file))

  def extractRgb3d(img: BufferedImage): Tensor[Double] = {
    val (w, h) = (img.getWidth, img.getHeight)
    val out = DenseMatrix.zeros[Double](3, w * h)
    (0 until w).foreach { x =>
      (0 until h).foreach { y =>
        val c = new Color(img.getRGB(x, y))
        val r = c.getRed   / 255.0
        val g = c.getGreen / 255.0
        val b = c.getBlue  / 255.0
        out.update(0, x * h + y, r)
        out.update(1, x * h + y, g)
        out.update(2, x * h + y, b)
      }
    }
    new RgbTensor[Double](w, h, out)
  }


  /**
    * Represents a RGB image, linearized into a full row
    * per color channel using column major.
    */
  class RgbTensor[V](width: Int, height: Int, override val matrix: DenseMatrix[V]) extends Tensor[V] {

    val projection: ((Int, Int, Int)) => (Int, Int) = K => (K._3, K._1 * height + K._2)

    def map(x: (Int, Int, Int))(f: V => V): Tensorish[(Int, Int, Int), V] = {
      val newMat = matrix.copy
      val (row, col) = projection(x._1, x._2, x._3)
      newMat.update(row, col, f(apply(x)))
      new RgbTensor(width, height, newMat)
    }

    def mapAll[T: ClassTag : Zero](f: V => T): Tensorish[(Int, Int, Int), T] = {
      val mapped = matrix.data.map(f)
      new RgbTensor(width, height, DenseMatrix.create(matrix.rows, matrix.cols, mapped))
    }

  }


  /**
    * Loads portable gray map as flat vector
    */

  def extractPgm(path: String): DenseVector[Double] = extractPgm(new File(path))

  def extractPgm(file: File): DenseVector[Double] = {
    val raw = Source.fromFile(file).getLines.drop(2).toArray // P2, width, height
    val max = raw.head.toDouble
    val img = raw.tail.flatMap(_.split(" ")).map(_.toDouble / max)
    DenseVector(img)
  }


  /**
    * Loads image from `file` or `path` and returns flattened sequence of pixels,
    * activated based on `selector` result
    */

  def extractBinary(path: String, selector: Int => Boolean): DenseVector[Double] = extractBinary(new File(path), selector)

  def extractBinary(file: File, selector: Int => Boolean): DenseVector[Double] = {
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
