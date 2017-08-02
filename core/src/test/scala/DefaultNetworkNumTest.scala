import neuroflow.core.Activator._
import neuroflow.core._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure
import shapeless._


/**
  * @author bogdanski
  * @since 02.08.17
  */
class DefaultNetworkNumTest extends Specification {

  def is: SpecStructure =
    s2"""

    This spec will test the gradients from DefaultNetwork by comparison of the derived values with
    the approximated ones.

      - Check the gradients                                $gradCheck

  """

  def gradCheck = {
    import neuroflow.core.FFN.WeightProvider.oneWeights
    import neuroflow.nets.DefaultNetwork._

    val layout =
        Input(2)           ::
        Hidden(3, Sigmoid) ::
        Hidden(4, Sigmoid) ::
        Hidden(5, Sigmoid) ::
        Hidden(6, Sigmoid) ::
        Hidden(5, Sigmoid) ::
        Hidden(4, Sigmoid) ::
        Hidden(3, Sigmoid) ::
        Output(2, Sigmoid) :: HNil

    val netA = Network(layout, Settings(learningRate = { case _ => 1.0 }, iterations = 1, approximation = Some(Approximation(1E-5))))
    val netB = Network(layout, Settings(learningRate = { case _ => 1.0 }, iterations = 1))

    val xs = Seq(Vector(1.0, 1.0))

    netA.train(xs, xs)
    netB.train(xs, xs)

    println(netA)
    println(netB)

    val tolerance = 1E-4

    val equal = netA.weights.zip(netB.weights).map {
      case (a, b) =>
        (a - b).forall { (w, v) =>
          println(s"dw($w) - approx(dw($w)): " + v.abs)
          v.abs < tolerance
        }
    }.reduce { (l, r) => l && r }

    if (equal) success else failure
  }

}
