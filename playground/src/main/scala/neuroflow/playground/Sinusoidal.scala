package neuroflow.playground

import neuroflow.core.Activator.Tanh
import neuroflow.core._
import neuroflow.core.FFN.WeightProvider._
import neuroflow.nets.DynamicNetwork.constructor
import shapeless._

import scala.annotation.tailrec

/**
  * @author bogdanski
  * @since 22.01.16
  */
object Sinusoidal {

   /*

        Here the goal is to predict the shape of sin(10*x).
        The net will be trained with the exact function values drawn from the interval [0.0 : 0.8],
        and the task is to continue (or predict) the next values from the interval ]0.8 : 4.0]
        based solely on learned history.

        Feel free to read this article for the full story:
          http://znctr.com/blog/time-series-prediction

    */

  def apply = {
    val fn = Tanh
    val group = 4
    val sets = Settings(verbose = true, learningRate = 10.0, precision = 1E-6, 500, specifics = Some(Map("τ" -> 0.25, "c" -> 0.25)))
    val net = Network(Input(3) :: Hidden(5, fn) :: Hidden(3, fn) :: Output(1, fn) :: HNil, sets)
    val sinusoidal = Range.Double(0.0, 0.8, 0.05).grouped(group).toList.map(i => i.map(k => (k, Math.sin(10 * k))))
    val xsys = sinusoidal.map(s => (s.dropRight(1).map(_._2), s.takeRight(1).map(_._2)))
    val xs = xsys.map(_._1)
    val ys = xsys.map(_._2)
    net.train(xs, ys)
    val initial = Range.Double(0.0, 0.15, 0.05).zipWithIndex.map(p => (p._1, xs.head(p._2)))
    val result = predict(net, xs.head, 0.15, initial)
    result.foreach(r => println(s"${r._1}, ${r._2}"))
  }

  @tailrec def predict[T <: FeedForwardNetwork](net: T, last: Seq[Double], i: Double,
                                                results: Seq[(Double, Double)]): Seq[(Double, Double)] = {
    if (i < 4.0) {
      val score = net.evaluate(last).head
      predict(net, last.drop(1) :+ score, i + 0.05, results :+ (i, score))
    } else results
  }

}
