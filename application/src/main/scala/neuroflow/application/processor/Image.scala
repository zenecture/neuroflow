package neuroflow.application.processor

import java.awt.Color
import java.io.File
import javax.imageio.ImageIO

import scala.io.Source

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Image {

  /**
    * Loads image from `file` or `path` and returns flattened sequence
    * of all color channels and pixels
    */
  def extractRgb(path: String): Vector[Double] = extractRgb(new File(path))
  def extractRgb(file: File): Vector[Double] = {
    val img = ImageIO.read(file)
    (0 until img.getHeight) flatMap { h =>
      (0 until img.getWidth) flatMap { w =>
        val c = new Color(img.getRGB(w, h))
        c.getRed / 255.0 :: c.getGreen / 255.0 :: c.getBlue / 255.0 :: Nil
      }
    }
  }.toVector

  def extractPgm(path: String): Vector[Double] = extractPgm(new File(path))
  def extractPgm(file: File): Vector[Double] = {
    val raw = Source.fromFile(file).getLines.drop(2).toVector // P2, width, height
    val max = raw.head.toDouble
    val img = raw.tail.flatMap(_.split(" ")).map(_.toDouble / max)
    img
  }

  /**
    * Loads image from `file` or `path` and returns flattened sequence of pixels,
    * activated based on `selector` result
    */
  def extractBinary(path: String, selector: Int => Boolean): Vector[Double] = extractBinary(new File(path), selector)
  def extractBinary(file: File, selector: Int => Boolean): Vector[Double] = {
    val img = ImageIO.read(file)
    (0 until img.getHeight).flatMap { h =>
      (0 until img.getWidth).flatMap { w =>
        val c = new Color(img.getRGB(w, h))
        (if (selector(c.getRed) || selector(c.getBlue) || selector(c.getGreen)) 1.0 else 0.0) :: Nil
      }
    }
  }.toVector

}
