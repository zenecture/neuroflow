package neuroflow.playground

import neuroflow.application.plugin.Style._
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.DynamicNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object XOR {

  def apply = {

    /*

       If you are new to neural nets and on the hunt for a
       rather informal blog post about the theory behind them:
         http://znctr.com/blog/artificial-neural-networks

     */

    val fn = Sigmoid
    val xs = -->(->(0.0, 0.0), ->(0.0, 1.0), ->(1.0, 0.0), ->(1.0, 1.0))
    val ys = -->(->(0.0), ->(1.0), ->(1.0), ->(0.0))
    val settings = Settings(verbose = true, learningRate = 100.0, precision = 0.00001,
      maxIterations = 20000, regularization = None, approximation = None, specifics = None)
    val net = Network(Input(2) :: Hidden(3, fn) :: Output(1, fn) :: HNil, settings)
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

//    println("Taking a lot of samples from model:")
//    Range.Double(0.0, 1.0, 0.01) map { x1 =>
//      Range.Double(0.0, 1.0, 0.01) map { x2 =>
//        println(s"$x1, $x2, ${net.evaluate(Seq(x1, x2)).head}")
//      }
//    }
  }

}
