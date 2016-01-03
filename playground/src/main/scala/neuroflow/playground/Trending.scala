package neuroflow.playground

import neuroflow.core.Activator.Sigmoid
import neuroflow.core.{Output, Hidden, Input, Network}

import scala.util.Random

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Trending {
  def apply = {

    def noise = if (Random.nextDouble > 0.5) 0.0625 else -0.0625

    // Training
    val trend = Range.Double(0.0, 1.1, 0.1).flatMap(i => Seq(i, i))
    val flat = Range.Double(0.0, 1.1, 0.1).flatMap( i => Seq(i, 0.3))

    //Testing
    val trendTest = Range.Double(0.0, 1.1, 0.1).flatMap(i => Seq(i, (1 + Random.nextDouble) * i)) // Linear trend with noise on slope
    val flatTest = Range.Double(0.0, 1.1, 0.1).flatMap(i => Seq(i, 0.3 + noise)) // Flat with additive noise
    val declineTest = Range.Double(0.0, 1.1, 0.1).flatMap(i => Seq(i, 1.0 - i)) // Linear decline trend
    val squareTest = Range.Double(0.0, 1.1, 0.1).flatMap(i => Seq(i, i * i)) // Square trend
    val squareDeclineTest = Range.Double(0.0, 1.1, 0.1).flatMap(i => Seq(i, (-1 * i * i) + 1.0)) // Square decline trend
    val sineTest = Range.Double(0.0, 1.1, 0.1).flatMap(i => Seq(i, 0.5*Math.sin(3*i))) // Jamming trend
    val cosineTest = Range.Double(0.0, 1.1, 0.1).flatMap(i => Seq(i, 0.5*Math.cos(6*i) + 0.5)) // Hero, to Zero, to Hero
    val upDownTest = Range.Double(0.0, 1.1, 0.1).flatMap(i => Seq(i, i + (Math.sin(100*i) / 3))) // Sinusoidal Noise


    val net = Network(Input(trend.size) :: Hidden(25, Sigmoid.apply) :: Output(1, Sigmoid.apply) :: Nil)
    net.train(Seq(trend, flat), Seq(Seq(1.0), Seq(0.0)), 20.0, 0.0001, 10000)

    val trendResult = net.evaluate(trendTest)
    val flatResult = net.evaluate(flatTest)
    val declineResult = net.evaluate(declineTest)
    val squareResult = net.evaluate(squareTest)
    val squareDeclineResult = net.evaluate(squareDeclineTest)
    val sineResult = net.evaluate(sineTest)
    val cosineResult = net.evaluate(cosineTest)
    val upDownResult = net.evaluate(upDownTest)

    println(s"Flat Result: $flatResult")
    println(s"Linear Trend Result: $trendResult")
    println(s"Square Trend Result: $squareResult")
    println(s"Linear Decline Trend Result: $declineResult")
    println(s"Square Decline Trend Result: $squareDeclineResult")
    println(s"Sine Result: $sineResult")
    println(s"Cosine Result: $cosineResult")
    println(s"UpDown Result: $upDownResult")
  }
}
