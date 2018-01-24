import java.net.URL
import javax.imageio.ImageIO

import neuroflow.application.processor.Image
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
      - Load a rgb image into a Matrix             $rgbMatrix

  """


  val image = new URL("http://znctr.com/new-landing/senchabg.jpg")

  def rgbMatrix = {
    val img = ImageIO.read(image)
    val vol = Image.extractRgb3d(image)
    if (img.getWidth * img.getHeight == vol.cols && vol.rows == 3) success else failure
  }

}
