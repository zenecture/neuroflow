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
      Map (
     // 0 -> input vector, no weights
        1 -> (0.0, 1.0),
        2 -> (0.0, 0.1)
      )
    }

    val f = Sigmoid

    val L = Vector(2)     ::
             Dense(3, f)  ::
             Dense(1, f)  ::  SquaredError()

    val net = Network(
      layout = L,
      settings = Settings[Double](
        learningRate = { case (_, _) => 1.0 },
        iterations = 100000,
        lossFuncOutput = Some(LossFuncOutput(Some("/Users/felix/github/unversioned/lossFunc.txt")))
      )
    )

    val xs = Seq(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
    val ys = Seq(->(0.0), ->(1.0), ->(1.0), ->(0.0))

    net.train(xs, ys)

    xs.foreach { x =>
      println(s"Input: $x, Output: ${net(x)}")
    }

    println("Network was: " + net)

  }

}


/*
      ...
    [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.09.2018 23:21:07:156] Iteration 99999.1, Avg. Loss = 0,000160901, Vector: 1.6090055545206475E-4
    [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.09.2018 23:21:07:156] Iteration 100000.1, Avg. Loss = 0,000160899, Vector: 1.6089888616269398E-4
    [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [03.09.2018 23:21:07:156] Took 100000 of 100000 iterations.
    Input: DenseVector(0.0, 0.0), Output: DenseVector(0.006936831967249095)
    Input: DenseVector(0.0, 1.0), Output: DenseVector(0.9909241417205932)
    Input: DenseVector(1.0, 0.0), Output: DenseVector(0.9892000479937961)
    Input: DenseVector(1.0, 1.0), Output: DenseVector(0.008640869703306806)
    Network was:
    ---
    -5.6134085682826855  -7.9952489974746195  -1.542845474986887
    11.18611554484063    -6.7407859286679646  0.9044785552871019
    ---
    -13.772289650351814
    -22.131232331989253
    25.97562379449903

 */

