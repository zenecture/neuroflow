package neuroflow.playground

import neuroflow.common.VectorTranslation._
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._
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
        based solely on learned history. This is achieved through a time window and self-feeding traversal.

        Feel free to read this article for the full story:
          http://znctr.com/blog/time-series-prediction

    */

  implicit val wp = FFN.WeightProvider(-0.2, 0.2)

  def apply = {

    val fn = Tanh
    val group = 4
    val sets = Settings(verbose = true, learningRate = { case (_, _) => 1E-1 }, precision = 1E-9, iterations = 1000)
    val net = Network(Input(3) :: Dense(5, fn) :: Dense(3, fn) :: Output(1, fn) :: HNil, sets)
    val sinusoidal = Range.Double(0.0, 0.8, 0.05).grouped(group).toVector.map(i => i.toVector.map(k => (k, Math.sin(10 * k))))
    val xsys = sinusoidal.map(s => (s.dropRight(1).map(_._2), s.takeRight(1).map(_._2)))
    val xs = xsys.map(_._1.dv)
    val ys = xsys.map(_._2.dv)
    net.train(xs, ys)
    val initial = Range.Double(0.0, 0.15, 0.05).zipWithIndex.map(p => (p._1, xs.head(p._2))).toVector
    val result = predict(net, xs.head.vv, 0.15, initial)
    result.foreach(r => println(s"${r._1}, ${r._2}"))

  }

  @tailrec def predict[T <: FeedForwardNetwork](net: T, last: Vector[Double], i: Double,
                                                results: Vector[(Double, Double)]): Vector[(Double, Double)] = {
    if (i < 4.0) {
      val score = net(last.dv)(0)
      predict(net, last.drop(1) :+ score, i + 0.05, results :+ (i, score))
    } else results
  }

}
