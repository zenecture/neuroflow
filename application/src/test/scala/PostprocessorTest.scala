import org.specs2.Specification
import org.specs2.specification.core.SpecStructure
import neuroflow.application.plugin.Style.->
import neuroflow.application.processor.Normalizer

/**
  * @author bogdanski
  * @since 17.06.16
  */
class ProcessorTest extends Specification {

  // IO race conditions will appear otherwise

  def is: SpecStructure =
    s2"""

    This spec will test IO related functionality.

    It should:
      - Normalize a vector to max = 1.0    $normalize

  """

  def normalize = {
    Normalizer(->(0.0, 0.50, 1.0, 1.50, 2.0)).max must equalTo(1.0)
    Normalizer(->(0.0, 0.50, 1.0, 1.50, 2.0)) must equalTo(->(0.0, 0.25, 0.5, 0.75, 1.0))
  }

}