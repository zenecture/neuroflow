package neuroflow.playground

import neuroflow.core.Activator.Tanh
import neuroflow.application.plugin.Style._
import neuroflow.core._
import neuroflow.core.WeightProvider.randomWeights
import neuroflow.nets.DynamicNetwork.constructor

/**
  * @author bogdanski
  * @since 22.01.16
  */
object Sinusoidal {

  def apply = {
    val sets = Settings(true, 10.0, 0.0001, 1000, None, None, Some(Map("t" -> 0.5, "c" -> 0.5)))
    val fn = Tanh.apply
    val net = Network(Input(1) :: Hidden(6, fn) :: Hidden(6, fn) :: Output(1, fn) :: Nil, sets)
    val sinusoidal = Range.Double(0.0, 1.0, 0.01).map(i => (i, Math.sin(10 * i)))
    val xs = sinusoidal.map(p => ->(p._1))
    val ys = sinusoidal.map(p => ->(p._2))
    net.train(xs, ys)

    val result = xs.map(p => (p.head, net.evaluate(p)))
    result.foreach { r =>
      println(s"${r._1}, ${r._2.head}")
    }
  }

}
