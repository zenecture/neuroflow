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

    implicit val weights = WeightBreeder[Double].normal(μ = 0.0, σ = 1.0)

    val f = Sigmoid

    val L = Vector (2)     ::
            Dense  (3, f)  ::
            Dense  (1, f)  ::  SquaredError()

    val net = Network(
      layout = L,
      settings = Settings[Double](
        learningRate = { case (_, _) => 1.0 },
        iterations = 100000
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
      [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [10.09.2018 19:52:02:739] Iteration 99998.1, Avg. Loss = 8,70932e-05, Vector: 8.709321657962622E-5
      [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [10.09.2018 19:52:02:739] Iteration 99999.1, Avg. Loss = 8,70923e-05, Vector: 8.709233203724416E-5
      [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [10.09.2018 19:52:02:739] Iteration 100000.1, Avg. Loss = 8,70914e-05, Vector: 8.709144751277962E-5
      [run-main-0] INFO neuroflow.nets.cpu.DenseNetworkDouble - [10.09.2018 19:52:02:740] Took 100000 of 100000 iterations.
      Input: DenseVector(0.0, 0.0), Output: DenseVector(0.010012593375718236)
      Input: DenseVector(0.0, 1.0), Output: DenseVector(0.9939939161874651)
      Input: DenseVector(1.0, 0.0), Output: DenseVector(0.9939952785050477)
      Input: DenseVector(1.0, 1.0), Output: DenseVector(0.001341408554410443)
      Network was:
      ---
      -4.803669356636045  7.023630858404076   5.881032948167752
      7.183252477792682   -4.750168900470756  5.907718336854065
      ---
      -14.456524517267937
      -14.463219452537524
      19.732046801403914
      [success] Total time: 16 s, completed 10.09.2018 19:52:02


 */

