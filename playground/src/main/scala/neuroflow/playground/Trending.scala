package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.common.VectorTranslation._
import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.cpu.DenseNetwork._
import shapeless._

import scala.util.Random

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Trending {

  /*

     Here the goal is to detect a trend in two-dimensional space (stock market, audio waves, ...)

     Feel free to read this article for the full story:
       http://znctr.com/blog/natural-trend-detection

   */

  def apply = {

    import neuroflow.application.plugin.Notation

    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].random(-1, 1)

    def noise = if (Random.nextDouble > 0.5) 0.0625 else -0.0625

    // Training
    val trend = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i)).toVector
    val flat = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.3)).toVector

    // Testing
    val trendTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, (1 + Random.nextDouble) * i)) // Linear trend with noise on slope
    val flatTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.3 + noise)) // Flat with additive noise
    val declineTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 1.0 - i)) // Linear decline trend
    val squareTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i * i)) // Square trend
    val squareDeclineTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, (-1 * i * i) + 1.0)) // Square decline trend
    val jammingTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.5 * Math.sin(3 * i))) // Jamming trend
    val heroZeroTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.5 * Math.cos(6 * i) + 0.5)) // Hero, to Zero, to Hero
    val oscillating = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, (Math.sin(100 * i) / 3) + 0.5)) // Oscillating
    val oscillatingUp = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i + (Math.sin(100 * i) / 3))) // Oscillating Up Trend
    val oscillatingDown = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, -i + (Math.sin(100 * i) / 3) + 1)) // Oscillating Down Trend
    val realWorld = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i + (Math.sin(100 * i) / 3) * Random.nextDouble)) // Real world data
    val random = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, Random.nextDouble)) // Random

    val fn = Sigmoid
    val settings = Settings[Double](learningRate = { case (_, _) => 0.5 }, precision = 1E-4, iterations = 10000)
    val net = Network(Input(trend.size) :: Dense(25, fn) :: Output(1, fn) :: HNil, settings)

    import Notation.Implicits.toVector

    net.train(Seq(trend.dv, flat.dv), Seq(->(1.0), ->(0.0)))

    println(s"Weights: ${net.weights.map(_.size).sum}")

    val trendResult = net.evaluate(trendTest)
    val flatResult = net.evaluate(flatTest)
    val declineResult = net.evaluate(declineTest)
    val squareResult = net.evaluate(squareTest)
    val squareDeclineResult = net.evaluate(squareDeclineTest)
    val jammingResult = net.evaluate(jammingTest)
    val heroZeroResult = net.evaluate(heroZeroTest)
    val oscillatingResult = net.evaluate(oscillating)
    val oscillatingUpResult = net.evaluate(oscillatingUp)
    val oscillatingDownResult = net.evaluate(oscillatingDown)
    val realWorldResult = net.evaluate(realWorld)
    val randomResult = net.evaluate(random)

    println(s"Flat Result: $flatResult")
    println(s"Linear Trend Result: $trendResult")
    println(s"Square Trend Result: $squareResult")
    println(s"Linear Decline Trend Result: $declineResult")
    println(s"Square Decline Trend Result: $squareDeclineResult")
    println(s"Jamming Result: $jammingResult")
    println(s"HeroZero Result: $heroZeroResult")
    println(s"Oscillating Result: $oscillatingResult")
    println(s"Oscillating Up Result: $oscillatingUpResult")
    println(s"Oscillating Down Result: $oscillatingDownResult")
    println(s"Real World Result: $realWorldResult")
    println(s"Random Result: $randomResult")

  }
}
