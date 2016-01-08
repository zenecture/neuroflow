package neuroflow.playground

import neuroflow.core.Activator.Sigmoid
import neuroflow.core._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object XOR {

  def apply = {
    val fn = Sigmoid.apply
    val network = Network(Input(2) :: Hidden(3, fn) :: Output(1, fn) :: Nil)
    network.train(Seq(Seq(0.0, 0.0), Seq(0.0, 1.0), Seq(1.0, 0.0), Seq(1.0, 1.0)),
      Seq(Seq(0.0), Seq(1.0), Seq(1.0), Seq(0.0)), 1.0, 0.001, 10000000)

    val a = network.evaluate(Seq(0.0, 0.0))
    val b = network.evaluate(Seq(0.0, 1.0))
    val c = network.evaluate(Seq(1.0, 0.0))
    val d = network.evaluate(Seq(1.0, 1.0))

    println(s"Input: 0.0, 0.0   Output: $a")
    println(s"Input: 0.0, 1.0   Output: $b")
    println(s"Input: 1.0, 0.0   Output: $c")
    println(s"Input: 1.0, 1.0   Output: $d")

    println("Network was: " + network)
  }

}
