package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.core.Activator._
import neuroflow.core._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object XOR {

  def apply = {

    /*

       The XOR-function is not linearly separable, thus we need
       something which naturally copes with non-linearities.

       ANNs to the rescue!

       If you are new to neural nets and on the hunt for a
       rather informal blog post about the theory behind them:
         http://znctr.com/blog/artificial-neural-networks

     */

    import neuroflow.nets.cpu.DenseNetwork._

    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].normal {
      Map ( // normal config per weight layer
        0 -> (0.0, 1.0),
        1 -> (0.0, 0.1)
      )
    }

    val fn = Sigmoid

    val xs = Seq(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
    val ys = Seq(->(0.0), ->(1.0), ->(1.0), ->(0.0))

    val settings = Settings[Double](
      learningRate = { case (_, _) => 1.0 },
      iterations = 100000,
      lossFuncOutput = Some(LossFuncOutput(Some("/Users/felix/github/unversioned/lossFunc.txt"), None)))

    val net = Network(Vector(2) :: Dense(3, fn) :: Dense(1, fn) :: SquaredMeanError(), settings)
    net.train(xs, ys)

    val a = net.evaluate(->(0.0, 0.0))
    val b = net.evaluate(->(0.0, 1.0))
    val c = net.evaluate(->(1.0, 0.0))
    val d = net.evaluate(->(1.0, 1.0))

    println(s"Input: 0.0, 0.0   Output: $a")
    println(s"Input: 0.0, 1.0   Output: $b")
    println(s"Input: 1.0, 0.0   Output: $c")
    println(s"Input: 1.0, 1.0   Output: $d")

    println("Network was: " + net)

  }

}
