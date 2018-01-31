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
      - Load a RGB image into a Tensor             $img2Tensor

  """


  val image = new URL("http://znctr.com/new-landing/senchabg.jpg")

  def img2Tensor = {
    val img = ImageIO.read(image)
    val tensor = Image.extractRgb3d(image)
    if (img.getWidth * img.getHeight == tensor.matrix.cols && tensor.matrix.rows == 3) success else failure
  }

}
