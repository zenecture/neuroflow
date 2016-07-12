package neuroflow.playground

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

    val stepSize = 0.1
    val a = Range.Double(-1.0, 0.0, stepSize).map(x => (->(sin(10 * x)), ->(-1.0, 1.0)))
    val b = Range.Double(0.0, 1.0, stepSize).map(x => (->(cos(3 * x)), ->(1.0, -1.0)))
    val all = a ++ b
    val f = Tanh
    val net = Network(Input(1) :: Hidden(3, f) :: Hidden(3, f) :: Output(2, f) :: HNil,
      Settings(iterations = 2000,
        learningRate = 0.2,
        partitions = Some(->(a.indexOf(a.last))),
        approximation = Some(Approximation(1E-9))))

    net.train(all.map(_._1), all.map(_._2))

    val resA = net.evaluateMean(a.map(_._1))
    println("sin(10x) classified as: " + resA)

    val resB = net.evaluateMean(b.map(_._1))
    println("cos(3x) classified as: " + resB)

  }

}
