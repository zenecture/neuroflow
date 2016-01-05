package neuroflow.application.classification

import java.awt.Color
import java.io.File
import javax.imageio.ImageIO

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Image {

  /**
    * Loads image from `file` or `path` and returns flattened sequence
    * of all color channels and pixels
    */
  def extractRgb(path: String): Seq[Double] = extractRgb(new File(path))
  def extractRgb(file: File): Seq[Double] = {
    val img = ImageIO.read(file)
    (0 to img.getWidth - 1) flatMap { w =>
      (0 to img.getHeight - 1) flatMap { h =>
        val c = new Color(img.getRGB(w, h))
        c.getRed / 255.0 :: c.getGreen / 255.0 :: c.getBlue / 255.0 :: Nil
      }
    }
  }

  /*
    TODO: provide option to flip width/height for optimal training
   */

  /**
    * Loads image from `file` or `path` and returns flattened sequence of pixels,
    * activated based on `selector` result
    */
  def extractBinary(path: String, selector: Int => Boolean): Seq[Double] = extractBinary(new File(path), selector)
  def extractBinary(file: File, selector: Int => Boolean): Seq[Double] = {
    val img = ImageIO.read(file)
    (0 to img.getHeight - 1) flatMap { h =>
      (0 to img.getWidth - 1) flatMap { w =>
        val c = new Color(img.getRGB(w, h))
        (if (selector(c.getRed) || selector(c.getBlue) || selector(c.getGreen)) 1.0 else 0.0) :: Nil
      }
    }
  }



}
