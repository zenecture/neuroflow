package neuroflow.playground

import neuroflow.core.Activator.Sigmoid
import neuroflow.core.WeightProvider.randomWeights
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._
import neuroflow.application.plugin.Style._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object XOR {

  def apply = {
    val fn = Sigmoid.apply
    val xs = -->(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
    val ys = -->(->(0.0), ->(1.0), ->(1.0), ->(0.0))
    val net = Network(Input(2) :: Hidden(3, fn) :: Output(1, fn) :: Nil)
    val trainSets = TrainSettings(stepSize = 2.0, precision = 0.001, maxIterations = 10000)
    net.train(xs, ys, trainSets)

    val a = net.evaluate(->(0.0, 0.0))
    val b = net.evaluate(->(0.0, 1.0))
    val c = net.evaluate(->(1.0, 0.0))
    val d = net.evaluate(->(1.0, 1.0))

    println(s"Input: 0.0, 0.0   Output: $a")
    println(s"Input: 0.0, 1.0   Output: $b")
    println(s"Input: 1.0, 0.0   Output: $c")
    println(s"Input: 1.0, 1.0   Output: $d")

    println("Network was: " + net)

    println("Taking alot of samples from model:")
    Range.Double(0.0, 1.0, 0.01) map { x1 =>
      Range.Double(0.0, 1.0, 0.01) map { x2 =>
        println(s"$x1, $x2, ${net.evaluate(Seq(x1, x2)).head}")
      }
    }
  }

}
