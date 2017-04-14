package neuroflow.playground

import neuroflow.application.plugin.Notation._
import neuroflow.core.Activator._
import neuroflow.core._
import shapeless._

import scala.math._

/**
  * @author bogdanski
  * @since 07.07.16
  */

object Sequences {

  def apply = {
    cosine2sineRNN
    linear2Step
    linear2cosinesine
    cosinesineClassification
    randomPointMapping
    randomPointClassification
  }

  /*
      The LSTM is able to learn the function cos(10x) -> sin(10x)
      as a sequence with input dimension = 1 (time step by time step).
   */

  def cosine2sineRNN = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-0.2, 0.2)

    val stepSize = 0.1
    val xsys = Range.Double(0.0, 1.0, stepSize).map(x => (->(cos(10 * x)), ->(sin(10 * x))))
    val f = Tanh
    val net = Network(Input(1) :: Hidden(5, f) :: Output(1, f) :: HNil,
      Settings(iterations = 5000, learningRate = 0.2, approximation = Some(Approximation(1E-9))))

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = net.evaluate(xsys.map(_._1))
    Range.Double(0.0, 1.0, stepSize).zip(res).foreach { case (l, r) => println(s"$l, ${r.head}") }

  }

  /*

      Simply learn to map from x -> { 0.5 | x < 0.8, 1.0 | >= 0.8 }

            it looks like this:  ________‾‾

  */

  def linear2Step = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-5.0, 5.0)

    val stepSize = 0.01
    val xsys = Range.Double(0.0, 1.0, stepSize).map(x => (->(x),->(if (x < 0.8) 0.5 else 1.0)))
    val f = Sigmoid
    val net = Network(Input(1) :: Hidden(3, f) :: Hidden(3, f) :: Output(1, f) :: HNil,
      Settings(iterations = 5000, learningRate = 0.2,
        approximation = Some(Approximation(1E-12)),
        errorFuncOutput = Some(ErrorFuncOutput(file = Some("/Users/felix/Downloads/class-out-3.txt")))))

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = net.evaluate(xsys.map(_._1))
    Range.Double(0.0, 1.0, stepSize).zip(res).foreach { case (l, r) => println(s"$l, ${r.head}") }

  }

  /*

      Simply learn to map from x -> (sin(10x), cos(10x))

   */

  def linear2cosinesine = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-1.0, 1.0)

    val stepSize = 0.1
    val xsys = Range.Double(0.0, 1.0, stepSize).map(x => (->(x), ->(sin(10 * x), cos(10 * x))))
    val f = Tanh
    val net = Network(Input(1) :: Hidden(7, f) :: Hidden(7, f) :: Output(2, f) :: HNil,
      Settings(iterations = 5000, learningRate = 0.5, approximation = Some(Approximation(1E-9))))

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = net.evaluate(xsys.map(_._1))
    Range.Double(0.0, 1.0, stepSize).zip(res).foreach { case (l, r) => println(s"$l, ${r.head}") }
    Range.Double(0.0, 1.0, stepSize).zip(res).foreach { case (l, r) => println(s"$l, ${r.tail.head}") }

  }

   /*

       Feeds the net with the input sequence sin(10x) from -1 to 0
       followed by cos(3x) from 0 to 1. The first sine wave gets class ->(-1, 1),
       the second cosine wave gets class ->(1, -1). The input is partitioned.
       The task is to infer the correct class for both waves.

   */

  def cosinesineClassification = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-1.0, 1.0)

    val stepSize = 0.01
    val a = Range.Double.inclusive(-1.0, 0.0, stepSize).map(x => (->(sin(10 * x)), ∞(2))).dropRight(1) :+ (->(sin(0.0)), ->(-1.0, 1.0))
    val b = Range.Double.inclusive(0.0, 1.0, stepSize).map(x => (->(cos(3 * x)), ∞(2))).dropRight(1) :+ (->(cos(3 * 1.0)), ->(1.0, -1.0))
    val all = a ++ b
    val f = Tanh
    val net = Network(Input(1) :: Hidden(3, f) :: Output(2, f) :: HNil,
      Settings(iterations = 500 ,
        learningRate = 0.2,
        partitions = Some(Set(a.indices.last)),
        approximation = Some(Approximation(1E-9))))

    net.train(all.map(_._1), all.map(_._2))

    val resA = net.evaluate(a.map(_._1)).last
    println("sin(10x) classified as: " + resA)

    val resB = net.evaluate(b.map(_._1)).last
    println("cos(3x) classified as: " + resB)

    /*

    [main] INFO neuroflow.nets.LSTMNetwork - [23.02.2017 00:49:42:592] Taking step 498 - Mean Error 0,00175740 - Error Vector 0.0018376699960378448  0.001677134755030868
    [main] INFO neuroflow.nets.LSTMNetwork - [23.02.2017 00:49:42:835] Taking step 499 - Mean Error 0,00175292 - Error Vector 0.0018329138945172479  0.0016729262855594948
    [main] INFO neuroflow.nets.LSTMNetwork - [23.02.2017 00:49:43:092] Took 500 iterations of 500 with Mean Error = 0,00175
    sin(10x) classified as: Vector(-0.9493479343178891, 0.9487560367367897)
    cos(3x) classified as: Vector(0.9669737809893943, -0.9733254272618534)

     */

  }


  /*

      Learn to map between sequences of random points ρ in 3-dimensional space.

  */

  def randomPointMapping = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-1.0, 1.0)

    val xs = (1 to 9) map (_ => ρ(3))
    val ys = (1 to 9) map (_ => ρ(3))

    val net = Network(Input(3) :: Hidden(6, Tanh) :: Output(3, Tanh) :: HNil,
      Settings(iterations = 2000,
        learningRate = 0.5,
        approximation = Some(Approximation(1E-9)),
        errorFuncOutput = Some(ErrorFuncOutput(file = Some("/Users/felix/Downloads/lstm.txt"))),
        partitions = Some(Π(3, 3))))

    net.train(xs, ys)

    val res = net.evaluate(xs)

    println(ys)
    println(res)

  }


  /*

      Learn to classify sequences of { random k-dimensional points ρ } of length n in c-dimensional space.

  */

  def randomPointClassification = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-1.0, 1.0)

    val (c, n, k) = (5, 5, 3)

    val all = (0 until c).flatMap { cc =>
      (0 until n).map { _ => (ρ(k), ζ(c).updated(cc, 1.0)) }
    }

    val f = Tanh
    val net = Network(Input(k) :: Hidden(10, f) :: Output(c, f) :: HNil,
      Settings(iterations = 2500,
        learningRate = 0.2,
        partitions = Some(Π(c, n)),
        errorFuncOutput = Some(ErrorFuncOutput(file = Some("/Users/felix/Downloads/pointClass.txt"))),
        approximation = Some(Approximation(1E-9))))

    net.train(all.map(_._1), all.map(_._2))

    all.map(_._1).grouped(c).zipWithIndex.foreach {
      case (cc, i) =>
        val r = net.evaluate(cc).last
        println("Output: " + r)
        println(s"=> Sequence $i classified as: " + r.indexOf(r.max))
    }

    /*

    [main] INFO neuroflow.nets.LSTMNetwork - [14.04.2017 21:37:43:334] Taking step 2498 - Mean Error 0,0199973 - Error Vector 0.020348676062929685  0.015328818670401966  0.007982609467666121  ... (5 total)
    [main] INFO neuroflow.nets.LSTMNetwork - [14.04.2017 21:37:43:800] Taking step 2499 - Mean Error 0,0199750 - Error Vector 0.019562583517121535  0.015703208667418915  0.0078807582526649  ... (5 total)
    [main] INFO neuroflow.nets.LSTMNetwork - [14.04.2017 21:37:44:275] Took 2500 iterations of 2500 with Mean Error = 0,0200
    Output: Vector(0.9983561787908275, 0.00549805210656058, -0.024722029600166257, 0.012735524753142598, -0.00959132263403632)
    => Sequence 0 classified as: 0
    Output: Vector(-0.0017782985398927835, 0.9938773198502886, -3.935141218154506E-4, -0.006980498594619799, -0.004807550594164794)
    => Sequence 1 classified as: 1
    Output: Vector(0.0023940193784388526, 0.0013108505102978343, 0.9992664895156691, 0.007636462978281285, -9.36923708285344E-4)
    => Sequence 2 classified as: 2
    Output: Vector(-0.0100093929191527, 0.007742858109792495, 5.854930380571284E-4, 0.9852257956323768, 0.0035237886209175113)
    => Sequence 3 classified as: 3
    Output: Vector(0.006617376032490632, 0.012248865250137905, 6.528306302873112E-4, 0.015044261312782317, 0.9992332431978462)
    => Sequence 4 classified as: 4

     */

  }

}
