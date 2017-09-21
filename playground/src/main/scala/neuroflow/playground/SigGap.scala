package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.core.Activator.Sigmoid
import neuroflow.core.FFN.WeightProvider._
import neuroflow.core._
import neuroflow.nets.cpu.DenseNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object SigGap {

  /*

      This is a shallow neural net.
      The closed form solution is -a +a or +a -a for both weights,
      so the goal is to find weights that are close to this exact shape.

  */

  def apply = {

    val settings = Settings(learningRate = { case (_, _) => 0.1 }, precision = 1E-20, iterations = 100000)
    val net = Network(Input(2) :: Output(1, Sigmoid) :: HNil, settings)
    net.train(Seq(->(0.3, 0.3)), Seq(->(0.5)))

    println("Output: " + net.evaluate(->(0.3, 0.3)))
    println("Parameters must roughly be of shape: -a, +a or +a, -a")
    println("Network was " + net)

  }

}
