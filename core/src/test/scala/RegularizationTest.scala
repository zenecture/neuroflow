import neuroflow.core.Activator.Linear
import neuroflow.core._
import neuroflow.core.FFN.WeightProvider.oneWeights
import neuroflow.nets.DefaultNetwork._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

import shapeless._

/**
  * @author bogdanski
  * @since 22.06.16
  */
class RegularizationTest extends Specification {

  def is: SpecStructure =
    s2"""

    This spec will test the regularization techniques.

    It should:
      - Check the early stopping logic                           $earlyStopping

  """

  def earlyStopping = {

    val (xs, ys) = (Seq(Seq(1.0), Seq(2.0), Seq(3.0)), Seq(Seq(3.2), Seq(5.8), Seq(9.2)))

    val net = Network(Input(1) :: Hidden(3, Linear) :: Output(1, Linear) :: HNil,
      Settings(regularization = Some(EarlyStopping(xs, ys, 0.8))))

    net.evaluate(Seq(1.0)) must be equalTo Seq(3.0)
    net.evaluate(Seq(2.0)) must be equalTo Seq(6.0)
    net.evaluate(Seq(3.0)) must be equalTo Seq(9.0)

    net.shouldStopEarly must be equalTo false
    net.shouldStopEarly must be equalTo true

  }

}
