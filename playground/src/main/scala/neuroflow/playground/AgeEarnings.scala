package neuroflow.playground


import neuroflow.core.Activator.Sigmoid
import neuroflow.core.FFN.WeightProvider._
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._
import shapeless._

import scala.io.Source

/**
  * @author bogdanski
  * @since 03.01.16
  */


object AgeEarnings {

  /*

     Here we compare Neural Net vs. Gaussian.
     Feel free to read this article for the full story:
        http://znctr.com/blog/gaussian-vs-neural-net

  */


  def apply = {

    val src = Source.fromInputStream(getClass.getClassLoader.getResourceAsStream("file/income.txt")).getLines().map(_.split(",")).flatMap(k => {
      (if (k.size > 14) Some(k(14)) else None).map { over50k => (k(0).toDouble, if (over50k.equals(" >50K")) 1.0 else 0.0) }
    }).toList

    val train = src.take(2000)
    //val test = src.drop(1000)
    val sets = Settings(verbose = true, learningRate = 0.05, precision = 0.001, iterations = 5000,
      regularization = None, approximation = None, specifics = None)
    val network = Network(Input(1) :: Hidden(20, Sigmoid) :: Output(1, Sigmoid) :: HNil, sets)
    val maxAge = train.map(_._1).sorted.reverse.head
    val xs = train.map(a => Seq(a._1 / maxAge))
    val ys = train.map(a => Seq(a._2))
    network.train(xs, ys)

    val allOver = src.filter(_._2 == 1.0)
    val ratio = allOver.size / src.size
    val mean = allOver.map(_._1).sum / allOver.size

    println(s"Mean of all $mean")
    println(s"Ratio $ratio")

    val result = Range.Double(0.0, 1.1, 0.01).map(k => (k * maxAge, network.evaluate(Seq(k))))
    val sum = result.map(_._2.head).sum
    println("Age, earning >50K")
    result.foreach { r => println(s"${r._1}, ${r._2.head * (1 / sum)}")}

  }


  /*
      After 5000 iterations the model predicted:

      Normalized to p(xi) * a, a = 1 / Σp(xi),
      such that Σp(xi) = 1:

      Age   P(Earning >50K)
      0.0,  0.000000287649
      9.0,  0.000071773252
      18.0, 0.005094161262
      27.0, 0.062065357723
      36.0, 0.168908028113
      45.0, 0.214381977708
      54.0, 0.197007855447
      63.0, 0.150627985683
      72.0, 0.101421685961
      81.0, 0.062946651639
      90.0, 0.037474235486

   */
}
