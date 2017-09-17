import java.net.URL

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
      - Load a 3d rgb volume and pad it             $rgbvolpad

  """


  val image = new URL("http://znctr.com/new-landing/senchabg.jpg")

  def rgbvolpad = {
    val vol = Image.extractRgb3d(image, dimension = None)
    val (w, h) = (vol.head.cols, vol.head.rows)
    val (dW, dH) = (w + 100, h + 200)
    val vol2 = Image.extractRgb3d(image, dimension = Some((dW, dH)))

    if (vol2.head.cols == dW && vol2.head.rows == dH) success else failure
  }

}
