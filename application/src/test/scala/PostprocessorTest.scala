import org.specs2.Specification
import org.specs2.specification.core.SpecStructure
import neuroflow.application.plugin.Notation.->
import neuroflow.application.processor.Normalizer

/**
  * @author bogdanski
  * @since 17.06.16
  */
class ProcessorTest extends Specification {

  def is: SpecStructure =
    s2"""

    This spec will test IO related functionality.

    It should:
      - Normalize a vector to max = 1.0    $max
      - Normalize a vector to unit vector  $unit

  """

  def max = {
    Normalizer.MaxUnit(->(0.0, 0.50, 1.0, 1.50, 2.0)).max must equalTo(1.0)
    Normalizer.MaxUnit(->(0.0, 0.50, 1.0, 1.50, 2.0)) must equalTo(->(0.0, 0.25, 0.5, 0.75, 1.0))
  }

  def unit = {
    Normalizer.UnitVector(->(2.0, 3.0, 4.0)).forall(_ <= 1.0) must beTrue
  }

}
