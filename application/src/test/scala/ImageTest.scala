import java.io.File
import java.net.URL
import javax.imageio.ImageIO

import neuroflow.application.processor.Image
import neuroflow.application.processor.Image._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

/**
  * @author bogdanski
  * @since 12.09.17
  */
class ImageTest extends Specification {

  sequential // IO race conditions will occur otherwise

  def is: SpecStructure =
    s2"""

    This spec will test image processor related functionality.

    It should:
      - Conversions with DenseVec                     $imageDenseVecConversions
      - Load a RGB image into a RgbTensor             $imageToRgbTensor
      - Convert a RgbTensor to Image                  $rgbTensorToImage
      - Scale an image                                $scaleImage

  """


  val image = new URL("http://znctr.com/new-landing/senchabg.jpg")

  def imageDenseVecConversions = {

    val imgA = ImageIO.read(image)
    val vec = Image.loadVectorRGB(image)
    val imgB = Image.imageFromVectorRGB(vec, imgA.getWidth, imgA.getHeight)

    Image.writeImage(imgB, "/Users/felix/github/unversioned/DenseVecToImage.jpg", JPG)

    (0 until imgA.getWidth).flatMap { x =>
      (0 until imgA.getHeight).map { y =>
        imgA.getRGB(x, y) == imgB.getRGB(x, y)
      }
    }.forall(_ == true) match {
      case true => success
      case false => failure
    }

  }

  def imageToRgbTensor = {
    val img = ImageIO.read(image)
    val tensor = Image.loadTensorRGB(image)
    if (img.getWidth * img.getHeight == tensor.matrix.cols && tensor.matrix.rows == 3) success else failure
  }

  def rgbTensorToImage = {

    val img1 = ImageIO.read(image)
    val tensor = Image.loadTensorRGB(img1)
    val img2 = Image.imageFromTensorRGB(tensor)

    Image.writeImage(img2, "/Users/felix/github/unversioned/RgbTensorToImage.jpg", JPG)

    if (img1.getWidth == img2.getWidth && img1.getHeight == img2.getHeight) success else failure

  }

  def scaleImage = {

    val img = ImageIO.read(image)
    val scaled = Image.scale(img, 200, 200)

    Image.writeImage(scaled, "/Users/felix/github/unversioned/scaleImage.jpg", JPG)

    if (scaled.getWidth == 200 && scaled.getHeight == 200) success else failure

  }

}

