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
      - Load a RGB image into a RgbTensor             $imageToRgbTensor
      - Convert a RgbTensor to Image                  $rgbTensorToImage

  """


  val image = new URL("http://znctr.com/new-landing/senchabg.jpg")

  def imageToRgbTensor = {
    val img = ImageIO.read(image)
    val tensor = Image.loadRgbTensor(image)
    if (img.getWidth * img.getHeight == tensor.matrix.cols && tensor.matrix.rows == 3) success else failure
  }

  def rgbTensorToImage = {

    val img1 = ImageIO.read(image)
    val tensor = Image.loadRgbTensor(img1)
    val img2 = Image.imageFromRgbTensor(tensor)

    Image.writeImage(img2, "/Users/felix/github/unversioned/RgbTensorToImage.jpg", JPG)

    if (img1.getWidth == img2.getWidth && img1.getHeight == img2.getHeight) success else failure

  }

}

