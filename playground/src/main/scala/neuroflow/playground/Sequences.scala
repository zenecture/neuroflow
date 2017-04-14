package neuroflow.playground

import neuroflow.application.plugin.Style
import neuroflow.application.plugin.Style._
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
    cosine2sineFFN
    cosine2sineRNN
    linear2Step
    linear2cosinesine
    cosinesineClassification
    randomPointMapping
  }

  def cosine2sineFFN = {

    /*
        The FFN will not be able to learn the function cos(10x) -> sin(10x)
        without using an input time window of higher kinded dimension (see Sinusoidal.scala).
            ("No need to learn what to store")
     */

    import neuroflow.nets.DefaultNetwork._
    implicit val wp = FFN.WeightProvider(-0.2, 0.2)

    val stepSize = 0.1
    val xsys = Range.Double(0.0, 1.0, stepSize).map(x => (->(cos(10 * x)), ->(sin(10 * x))))
    val f = Tanh
    val net = Network(Input(1) :: Hidden(10, f) :: Hidden(10, f) :: Hidden(10, f) :: Output(1, f) :: HNil,
      Settings(iterations = 2000, learningRate = 0.1))

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = xsys.map(_._1).map(net.evaluate)
    Range.Double(0.0, 1.0, stepSize).zip(res).foreach { case (l, r) => println(s"$l, ${r.head}") }

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
       followed by cos(3x) from 0 to 1. The first sine wave gets class ->(1, -1),
       the second cosine wave gets class ->(-1, 1). The input is partitioned.
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

      Learn to map between random sequences points in 3-dimensional space.

  */

  def randomPointMapping = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-1.0, 1.0)

    val xs = (1 to 9) map (_ => Style.random(3))
    val ys = (1 to 9) map (_ => Style.random(3))

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

}
