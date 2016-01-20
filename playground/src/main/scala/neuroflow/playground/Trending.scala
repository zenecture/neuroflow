package neuroflow.playground

import neuroflow.core.Activator.Sigmoid
import neuroflow.core.WeightProvider.randomWeights
import neuroflow.core._
import neuroflow.nets.DefaultNetwork._

import scala.util.Random

/**
  * @author bogdanski
  * @since 03.01.16
  */
object Trending {
  def apply = {

    def noise = if (Random.nextDouble > 0.5) 0.0625 else -0.0625

    // Training
    val trend = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i))
    val flat = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.3))

    //Testing
    val trendTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, (1 + Random.nextDouble) * i)) // Linear trend with noise on slope
    val flatTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.3 + noise)) // Flat with additive noise
    val declineTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 1.0 - i)) // Linear decline trend
    val squareTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i * i)) // Square trend
    val squareDeclineTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, (-1 * i * i) + 1.0)) // Square decline trend
    val jammingTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.5*Math.sin(3*i))) // Jamming trend
    val heroZeroTest = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, 0.5*Math.cos(6*i) + 0.5)) // Hero, to Zero, to Hero
    val oscillating = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, (Math.sin(100*i) / 3) + 0.5)) // Oscillating
    val oscillatingUp = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i + (Math.sin(100*i) / 3))) // Oscillating Up Trend
    val oscillatingDown = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, -i + (Math.sin(100*i) / 3) + 1)) // Oscillating Down Trend
    val realWorld = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, i + (Math.sin(100*i) / 3) * Random.nextDouble)) // Real world data
    val random = Range.Double(0.0, 1.0, 0.01).flatMap(i => Seq(i, Random.nextDouble)) // Random


    val fn = Sigmoid.apply
    val net = Network(Input(trend.size) :: Hidden(25, fn) :: Output(1, fn) :: Nil)
    val trainSets = TrainSettings(learningRate = 20.0, precision = 0.0001, maxIterations = 10000, regularization = None)
    net.train(Seq(trend, flat), Seq(Seq(1.0), Seq(0.0)), trainSets)

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
