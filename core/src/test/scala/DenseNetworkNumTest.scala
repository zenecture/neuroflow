package neuroflow.nets

import breeze.linalg._
import neuroflow.core.Activator._
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

    import neuroflow.core.WeightProvider.ffn_double.fullyConnected
    import neuroflow.core.WeightProvider.normalSeed
    
    val f = Tanh

    val layout =
         Input(2)      ::
         Dense(7, f)   ::
         Dense(8, f)   ::
         Dense(7, f)   ::
        Output(2, f)   ::  HNil

    val rand = fullyConnected(layout.toList, normalSeed[Double](0.1, 0.1))

    implicit val wp = new WeightProvider[Double] {
      def apply(layers: Seq[Layer]): Weights[Double] = rand.map(_.copy)
    }

    val debuggableA = Debuggable[Double]()
    val debuggableB = Debuggable[Double]()


    val settings = Settings[Double](
      lossFunction = SquaredMeanError(),
      prettyPrint = true,
      learningRate = { case (_, _) => 1.0 },
      iterations = 1
    )

    val netA = Network(layout, settings.copy(updateRule = debuggableA))
    val netB = Network(layout, settings.copy(updateRule = debuggableB, approximation = Some(FiniteDifferences(1E-5))))

    val xs = Seq(DenseVector(0.5, 0.5), DenseVector(0.7, 0.7))
    val ys = Seq(DenseVector(1.0, 0.0), DenseVector(0.0, 1.0))

    netA.train(xs, ys)
    netB.train(xs, ys)

    val tolerance = 1E-7

    val equal = debuggableA.lastGradients.zip(debuggableB.lastGradients).map {
      case ((i, a), (_, b)) =>
        (a - b).forall { (w, v) =>
          val e = v.abs
          if (e == 0.0) true else {
            val x = debuggableA.lastGradients(i)(w)
            val y = debuggableB.lastGradients(i)(w)
            val m = math.max(x.abs, y.abs)
            val r = e / m
            if (r >= tolerance) {
              println(s"i = $i")
              println(s"e = $e")
              println(s"$r >= $tolerance")
              false
            } else true
          }
        }
    }.reduce { (l, r) => l && r }

    if (equal) success else failure

  }

}
