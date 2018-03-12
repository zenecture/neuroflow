package neuroflow.application.processor

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.net.URL
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero
import neuroflow.common.Logs
import neuroflow.core.Tensor3D

import scala.io.Source
import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Image extends Logs {


  /**
    * Loads image from `file` or `path` and returns flattened [[DenseVector]]
    * where pixel values are normalized to be <= 1.0.
    */

  def loadRgbVector(path: String): DenseVector[Double] = loadRgbVector(new File(path))

  def loadRgbVector(file: File): DenseVector[Double] = {
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

  def loadRgbTensor(url: URL): RgbTensor[Double] = loadRgbTensor(ImageIO.read(url))

  def loadRgbTensor(path: String): RgbTensor[Double] = loadRgbTensor(new File(path))

  def loadRgbTensor(file: File): RgbTensor[Double] = loadRgbTensor(ImageIO.read(file))

  def loadRgbTensor(img: BufferedImage): RgbTensor[Double] = {
    val (w, h) = (img.getWidth, img.getHeight)
    val out = DenseMatrix.zeros[Double](3, w * h)
    val tensor = new RgbTensor[Double](w, h, out)
    (0 until w).foreach { x =>
      (0 until h).foreach { y =>
        val c = new Color(img.getRGB(x, y))
        val rgb = c.getRed / 255.0 :: c.getGreen / 255.0 :: c.getBlue  / 255.0 :: Nil
        (0 until 3).foreach { z =>
          val (row, col) = tensor.projection(x, y, z)
          out.update(row, col, rgb(z))
        }
      }
    }
    tensor
  }


  /**
    * Represents a RGB image, accessible by (x, y, z) coordinates.
    * Where x, y are width, height and z is the color channel.
    */
  class RgbTensor[V](width: Int, height: Int, override val matrix: DenseMatrix[V]) extends Tensor3D[V] {

    val stride: Int = height

    def mapAt(x: (Int, Int, Int))(f: V => V): RgbTensor[V] = {
      val newMat = matrix.copy
      val (row, col) = projection(x._1, x._2, x._3)
      newMat.update(row, col, f(apply(x)))
      new RgbTensor(width, height, newMat)
    }

    def mapAll[T: ClassTag : Zero](f: V => T): RgbTensor[T] = {
      val mapped = matrix.data.map(f)
      new RgbTensor(width, height, DenseMatrix.create(matrix.rows, matrix.cols, mapped))
    }

  }


  /**
    * Loads portable gray map as flattened [[DenseVector]].
    */

  def loadPgm(path: String): DenseVector[Double] = loadPgm(new File(path))

  def loadPgm(file: File): DenseVector[Double] = {
    val raw = Source.fromFile(file).getLines.drop(2).toArray // P2, width, height
    val max = raw.head.toDouble
    val img = raw.tail.flatMap(_.split(" ")).map(_.toDouble / max)
    DenseVector(img)
  }


  /**
    * Loads image from `file` or `path` and returns flattened [[DenseVector]],
    * where pixels are white or black, depending on the `selector`.
    */
  def loadBinary(path: String, selector: Int => Boolean): DenseVector[Double] = loadBinary(new File(path), selector)

  def loadBinary(file: File, selector: Int => Boolean): DenseVector[Double] = {
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
