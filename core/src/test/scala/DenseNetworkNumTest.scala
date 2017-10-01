package neuroflow.nets

import breeze.linalg._
import neuroflow.core.Activator._
import neuroflow.core.WeightProvider.Double.FFN.{fullyConnected, randomD}
import neuroflow.core.Network.Weights
import neuroflow.core._
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure
import shapeless._


/**
  * @author bogdanski
  * @since 02.08.17
  */
class DenseNetworkNumTest extends Specification {

  def is: SpecStructure =
    s2"""

    This spec will test the gradients from DenseNetwork by
    comparing the analytical gradients with the approximated ones (finite diffs).

      - Check the gradients on CPU                       $gradCheckCPU
      - Check the gradients on GPU                       $gradCheckGPU

  """

  def gradCheckCPU = {

    import neuroflow.nets.cpu.DenseNetwork._
    check()

  }

  def gradCheckGPU = {

    import neuroflow.nets.gpu.DenseNetwork._
    check()

  }

  def check[Net <: FFN[Double]]()(implicit net: Constructor[Double, Net]) = {
    
    val f = Tanh

    val layout =
        Input(2)      ::
        Dense(7, f)   ::
        Dense(8, f)   ::
        Dense(7, f)   ::
        Output(2, f)  :: HNil

    val rand = fullyConnected(layout.toList, randomD(-1, 1))

    implicit val wp = new WeightProvider[Double] {
      def apply(layers: Seq[Layer]): Weights[Double] = rand.map(_.copy)
    }

    val debuggableA = Debuggable[Double]()
    val debuggableB = Debuggable[Double]()

    val netA = Network(layout, Settings(prettyPrint = true, learningRate = { case (_, _) => 1.0 }, updateRule = debuggableA, iterations = 1, approximation = Some(Approximation(1E-5))))
    val netB = Network(layout, Settings(prettyPrint = true, learningRate = { case (_, _) => 1.0 }, updateRule = debuggableB, iterations = 1))

    val xs = Seq(DenseVector(0.5, 0.5), DenseVector(1.0, 1.0))

    println(netA)
    println(netB)

    netA.train(xs, xs)
    netB.train(xs, xs)

    println(netA)
    println(netB)

    val tolerance = 1E-7

    val equal = debuggableA.lastGradients.zip(debuggableB.lastGradients).map {
      case ((i, a), (_, b)) =>
        println(s"i = $i")
        (a - b).forall { (w, v) =>
          val e = v.abs
          if (e == 0.0) {
            println(s"e = $e     ( a($w) = ${a(w)}, b($w) = ${b(w)} )")
            true
          } else {
            val x = debuggableA.lastGradients(i)(w)
            val y = debuggableB.lastGradients(i)(w)
            val m = math.max(x.abs, y.abs)
            val r = e / m
            println(s"e = $e")
            println(s"r = $r")
            r < tolerance
          }
        }
    }.reduce { (l, r) => l && r }

    if (equal) success else failure

  }

}
