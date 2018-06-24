package neuroflow.application.processor

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.net.URL

import javax.imageio.ImageIO
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero
import neuroflow.common.Logs
import neuroflow.core.{Tensor3D, Tensor3DImpl}

import scala.io.Source
import scala.reflect.ClassTag

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Image extends Logs {


  /**
    * Loads image from `url`, `file` or `path` and returns flattened [[DenseVector]]
    * where pixel values are normalized to be <= 1.0.
    */

  def loadVectorRGB(url: URL): DenseVector[Double] = loadVectorRGB(ImageIO.read(url))

  def loadVectorRGB(path: String): DenseVector[Double] = loadVectorRGB(new File(path))

  def loadVectorRGB(file: File): DenseVector[Double] = loadVectorRGB(ImageIO.read(file))

  def loadVectorRGB(img: BufferedImage): DenseVector[Double] = {
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
    * Represents a RGB image, accessible by (x, y, z) coordinates.
    * Where x, y are width, height and z is the color channel.
    */
  class TensorRGB[V](width: Int, height: Int, override val matrix: DenseMatrix[V]) extends Tensor3DImpl[V](matrix, width, height, 3) {
    override def mapAll[T: ClassTag : Zero](f: V => T): TensorRGB[T] = {
      new TensorRGB[T](width, height, matrix.map(f))
    }
  }


  /**
    * Loads image from `url`, `path` or `file` and returns a [[TensorRGB]].
    * All pixel colors are scaled from [0, 255] to [0.0, 1.0].
    */

  def loadTensorRGB(url: URL): TensorRGB[Double] = loadTensorRGB(ImageIO.read(url))

  def loadTensorRGB(path: String): TensorRGB[Double] = loadTensorRGB(new File(path))

  def loadTensorRGB(file: File): TensorRGB[Double] = loadTensorRGB(ImageIO.read(file))

  def loadTensorRGB(img: BufferedImage): TensorRGB[Double] = {
    val (w, h) = (img.getWidth, img.getHeight)
    val out = DenseMatrix.zeros[Double](3, w * h)
    val tensor = new TensorRGB[Double](w, h, out)
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

  sealed trait ImageFormat
  object PNG extends ImageFormat
  object JPG extends ImageFormat

  /**
    * Writes `img` to `filePath`.
    */
  def writeImage(img: BufferedImage, filePath: String, imageFormat: ImageFormat): Unit = {
    val outputfile = new File(filePath)
    imageFormat match {
      case PNG =>
        ImageIO.write(img, "png", outputfile)
      case JPG =>
        ImageIO.write(img, "jpg", outputfile)
    }
  }


  /**
    * Loads RGB image from [[DenseVector]] `v`.
    */
  def imageFromVectorRGB(v: DenseVector[Double], X: Int, Y: Int): BufferedImage = {
    val img = new BufferedImage(X, Y, BufferedImage.TYPE_INT_RGB)
    (0 until X).foreach { x =>
      (0 until Y).foreach { y =>
        val i = y * (X * 3) + (x * 3)
        val r = (v(i) * 255.0).toInt
        val g = (v(i + 1) * 255.0).toInt
        val b = (v(i + 2) * 255.0).toInt
        var rgb = r
        rgb = (rgb << 8) + g
        rgb = (rgb << 8) + b
        img.setRGB(x, y, rgb)
      }
    }
    img
  }


  /**
    * Loads image from [[TensorRGB]] `t`.
    * All pixel colors are scaled from [0.0, 1.0] to [0, 255].
    */
  def imageFromTensorRGB(t: TensorRGB[Double]): BufferedImage = {

    val img = new BufferedImage(t.X, t.Y, BufferedImage.TYPE_INT_RGB)

    (0 until t.X).foreach { x =>
      (0 until t.Y).foreach { y =>
        val r = (t(x, y, 0) * 255.0).toInt
        val g = (t(x, y, 1) * 255.0).toInt
        val b = (t(x, y, 2) * 255.0).toInt
        var rgb = r
        rgb = (rgb << 8) + g
        rgb = (rgb << 8) + b
        img.setRGB(x, y, rgb)
      }
    }

    img

  }


  /**
    * Extracts grayscale images from [[Tensor3D]] `t` by z-dimension.
    * The luminance can be amplified by `boost`.
    */
  def imagesFromTensor3D(t: Tensor3D[Double], boost: Double = 1.0): Seq[BufferedImage] = {

    val max = t.matrix.data.max

    (0 until t.matrix.rows).map { r =>
      val img = new BufferedImage(t.X, t.Y, BufferedImage.TYPE_INT_RGB)
      (0 until t.X).foreach { x =>
        (0 until t.Y).foreach { y =>
          val v = (t(x, y, r) / max * 255.0 * boost).toInt
          var rgb = v
          rgb = (rgb << 8) + v
          rgb = (rgb << 8) + v
          img.setRGB(x, y, rgb)
        }
      }
      img
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


