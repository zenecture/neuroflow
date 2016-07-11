package neuroflow.playground

import neuroflow.application.plugin.Style._
import neuroflow.core.Activator._
import neuroflow.core._
import shapeless.HNil

import scala.math._

/**
  * @author bogdanski
  * @since 07.07.16
  */

object Sequences {

  def apply = {
//    cosine2sineFFN
//    cosine2sineRNN
    linearToStep
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
    val xsys = Range.Double(0.0, 1.0, stepSize).map(s => (->(cos(10 * s)), ->(sin(10 * s))))
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
    val xsys = Range.Double(0.0, 1.0, stepSize).map(s => (->(cos(10 * s)), ->(sin(10 * s))))
    val f = Tanh
    val net = Network(Input(1) :: Hidden(5, f) :: Output(1, f) :: HNil,
      Settings(iterations = 5000, learningRate = 0.2, approximation = Some(Approximation(1E-9))))

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = net.evaluate(xsys.map(_._1))
    Range.Double(0.0, 1.0, stepSize).zip(res).foreach { case (l, r) => println(s"$l, ${r.head}") }

  }

  def linearToStep = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-2.0, 2.0)

    val stepSize = 0.01
    val xsys = Range.Double(0.0, 1.0, stepSize).map(s => (->(s),->(if (s < 0.8) 0.5 else 1.0)))
    val f = Sigmoid
    val net = Network(Input(1) :: Hidden(3, f) :: Hidden(3, f) :: Hidden(3, f) :: Output(1, f) :: HNil,
      Settings(iterations = 4000, learningRate = 0.2,
        approximation = Some(Approximation(1E-9)),
        errorFuncOutput = Some(ErrorFuncOutput(file = Some("/Users/felix/Downloads/class-out-3.txt")))))

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = net.evaluate(xsys.map(_._1))
    Range.Double(0.0, 1.0, stepSize).zip(res).foreach { case (l, r) => println(s"$l, ${r.head}") }

  }

}
