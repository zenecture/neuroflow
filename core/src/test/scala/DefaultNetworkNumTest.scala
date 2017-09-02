package neuroflow.nets

import neuroflow.core.Activator._
import neuroflow.core._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure
import shapeless._
import neuroflow.common.VectorTranslation._
import neuroflow.core
import neuroflow.core.FFN.{fullyConnected, random}
import neuroflow.core.Network.Weights


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
    import neuroflow.nets.DefaultNetwork._
    
    val f = Sigmoid

    val layout =
        Input(2)     ::
        Hidden(3, f) ::
        Hidden(4, f) ::
        Hidden(3, f) ::
        Output(2, f) :: HNil

    val rand = fullyConnected(layout.toList, random(-1, 1))

    implicit val wp = new WeightProvider {
      def apply(layers: Seq[Layer]): Weights = rand.map(_.copy)
    }

    val netA = Network(layout, Settings(learningRate = { case _ => 1.0 }, iterations = 1, approximation = Some(Approximation(1E-5))))
    val netB = Network(layout, Settings(learningRate = { case _ => 1.0 }, iterations = 1))

    val xs = Seq(Vector(0.5, 0.5).dv, Vector(1.0, 1.0).dv)

    println(netA)
    println(netB)

    netA.train(xs, xs)
    netB.train(xs, xs)

    println(netA)
    println(netB)

    val tolerance = 1E-1

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
