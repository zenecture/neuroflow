package neuroflow.playground

import neuroflow.core.Activator.Sigmoid
import neuroflow.core.WeightProvider.randomWeights
import neuroflow.core.{Input, Network, Output}
import neuroflow.nets.DefaultNetwork._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object SigGap {

  def apply = {
    val net = Network(Input(2) :: Output(1, Sigmoid.apply) :: Nil)
    net.train(Seq(Seq(0.3, 0.3)), Seq(Seq(0.5)), 0.1, 0.00000000000000000001, 100000)

    println("Output: " + net.evaluate(Seq(0.3, 0.3)))
    println("Parameters must be of kind: -a, +a or +a, -a")
    println("Network was " + net)
  }

}
