package neuroflow.nets

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.core.Activator._
import neuroflow.core.CNN.{convoluted, random}
import neuroflow.core.Network.Weights
import neuroflow.core._
import neuroflow.nets.ConvNetwork._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure
import shapeless._


/**
  * @author bogdanski
  * @since 02.09.17
  */
class ConvNetworkNumTest  extends Specification {

  def is: SpecStructure =
    s2"""

    This spec will test the gradients from ConvNetwork by comparison of the derived values with
    the approximated ones.

      - Check the gradients                                $gradCheck

  """

  def gradCheck = {

    val dim = (20, 20, 3)
    val out = 2

    val f = ReLU

    val debuggableA = Debuggable()
    val debuggableB = Debuggable()

    val a = Convolution(dimIn = dim,      field = 4, filters = 4, stride = 2, padding = 0, f)
    val b = Convolution(dimIn = a.dimOut, field = 4, filters = 2, stride = 1, padding = 0, f)

    val convs = a :: b :: HNil
    val fullies = Output(out, f) :: HNil

    val layout = convs ::: fullies

    val rand = convoluted(layout.toList, random(0.01, 0.1))

    implicit val wp = new WeightProvider {
      def apply(layers: Seq[Layer]): Weights = rand.map(_.copy)
    }

    val settings = Settings(
      prettyPrint = true,
      approximation = None,
      learningRate = { case _ => 1.0 },
      iterations = 1
    )

    val netA = Network(layout, settings.copy(updateRule = debuggableA))
    val netB = Network(layout, settings.copy(updateRule = debuggableB, approximation = Some(Approximation(1E-5))))

    val m = DenseMatrix.rand[Double](dim._1, dim._2)
    val n = DenseVector.rand[Double](out)

    val xs = Seq((1 to dim._3).map(_ => m))
    val ys = Seq(n)

    println(netA)
    println(netB)

    netA.train(xs, ys)
    netB.train(xs, ys)

    println(netA)
    println(netB)

    val tolerance = 1E-7

    val equal = debuggableA.lastGradients.zip(debuggableB.lastGradients).map {
      case ((i, a), (_, b)) =>
        println("i " + i)
        (a - b).forall { (w, v) =>
          val e = v.abs
          val x = debuggableA.lastGradients(i)(w)
          val y = debuggableB.lastGradients(i)(w)
          val m = math.max(x.abs, y.abs)
          val r = e / m.abs
          println(s"e = $e")
          println(s"r = $r")
          r < tolerance
        }
    }.reduce { (l, r) => l && r }

    if (equal) success else failure

  }

}
