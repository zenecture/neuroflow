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

    val L = Vector(2)    ::
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


