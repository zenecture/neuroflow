package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.core.Activators.Double._
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.nets.cpu.DenseNetwork._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object XOR {

  def apply = {

    /*

       The XOR-function is linearly not separable, so we need
       something which naturally copes with non-linearities.

       ANNs to the rescue!

       If you are new to neural nets and on the hunt for a
       rather informal blog post about the theory behind them:
         http://znctr.com/blog/artificial-neural-networks

     */


    implicit val weights = WeightBreeder[Double].normal {
      Map ( // normal config per layer index
        1 -> (0.0, 1.0),
        2 -> (0.0, 0.1)
      )
    }

    val fn = Sigmoid

    val xs = Seq(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
    val ys = Seq(->(0.0), ->(1.0), ->(1.0), ->(0.0))

    val settings = Settings[Double](
      learningRate = { case (_, _) => 1.0 },
      iterations = 100000,
      lossFuncOutput = Some(LossFuncOutput(Some("/Users/felix/github/unversioned/lossFunc.txt"), None)))


    val L = Vector(2) :: Dense(3, fn) :: Dense(1, fn) :: SquaredError()
    val net = Network(layout = L, settings)

    net.train(xs, ys)

    val a = net.apply(->(0.0, 0.0))
    val b = net.apply(->(0.0, 1.0))
    val c = net.apply(->(1.0, 0.0))
    val d = net.apply(->(1.0, 1.0))

    println(s"Input: 0.0, 0.0   Output: $a")
    println(s"Input: 0.0, 1.0   Output: $b")
    println(s"Input: 1.0, 0.0   Output: $c")
    println(s"Input: 1.0, 1.0   Output: $d")

    println("Network was: " + net)

  }

}

