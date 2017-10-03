package neuroflow.nets

import breeze.linalg.{DenseMatrix, DenseVector}
import neuroflow.core.Activator._
import neuroflow.core.Network.Weights
import neuroflow.core._

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

    This spec will test the gradients from ConvNetwork by
    comparing the analytical gradients with the approximated ones (finite diffs).

      - Check the gradients on CPU                       $gradCheckCPU
      - Check the gradients on GPU                       $gradCheckGPU

  """

  def gradCheckCPU = {

    import neuroflow.nets.cpu.ConvNetwork._
    check()

  }

  def gradCheckGPU = {

    import neuroflow.nets.gpu.ConvNetwork._
    check()

  }

  def check[Net <: CNN[Double]]()(implicit net: Constructor[Double, Net]) = {

    import neuroflow.core.WeightProvider.Double.CNN.{convoluted, normalD}

    val dim = (50, 25, 2)
    val out = 2

    val f = ReLU

    val debuggableA = Debuggable[Double]()
    val debuggableB = Debuggable[Double]()

    val a = Convolution(dimIn = dim,      padding = (4, 4), field = (6, 3), stride = (2, 2), filters = 2, activator = f)
    val b = Convolution(dimIn = a.dimOut, padding = (2, 2), field = (2, 4), stride = (1, 1), filters = 2, activator = f)
    val c = Convolution(dimIn = b.dimOut, padding = (1, 1), field = (1, 1), stride = (1, 2), filters = 2, activator = f)

    val convs = a :: b :: c :: HNil
    val fullies = Output(out, f) :: HNil

    val layout = convs ::: fullies

    val rand = convoluted(layout.toList, normalD(0.0, 0.1))

    implicit val wp = new WeightProvider[Double] {
      def apply(layers: Seq[Layer]): Weights[Double] = rand.map(_.copy)
    }

    val settings = Settings[Double](
      prettyPrint = true,
      approximation = None,
      lossFunction = SquaredMeanError(),
      learningRate = { case (_, _) => 1.0 },
      iterations = 1
    )

    val netA = Network(layout, settings.copy(updateRule = debuggableA))
    val netB = Network(layout, settings.copy(updateRule = debuggableB, approximation = Some(Approximation(1E-5))))

    val m = DenseMatrix.rand[Double](dim._1, dim._2)
    val n = DenseVector.zeros[Double](out)
    n.update(0, 1.0)

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
              println(s"r = $r")
              false
            } else true
          }
        }
    }.reduce { (l, r) => l && r }

    if (equal) success else failure

  }

}
