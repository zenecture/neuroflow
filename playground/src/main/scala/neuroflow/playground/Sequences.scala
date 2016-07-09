package neuroflow.playground

import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.application.plugin.Style._
import shapeless.HNil

import scala.math._

/**
  * @author bogdanski
  * @since 07.07.16
  */

object Sequences {

  /**
    *
    */
  def sinusoidalFFN = {

    import neuroflow.nets.DefaultNetwork._
    implicit val wp = FFN.WeightProvider(-0.2, 0.2)

    val stepSize = 0.1
    val xsys = Range.Double(0.0, 1.0, stepSize).map(s => (->(s), ->(sin(10 * s))))
    val f = Tanh
    val net = Network(Input(1) :: Hidden(10, f) :: Hidden(10, f) :: Hidden(10, f) :: Output(1, f) :: HNil,
      Settings(iterations = 2000, learningRate = 2.0))

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = xsys.map(_._1).map(net.evaluate)
    xsys.map(_._1).zip(res).foreach { case (l, r) => println(s"${l.head}, ${r.head}") }

  }

  def sinusoidalRNN = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-0.2, 0.2)

    val stepSize = 0.05
    val xsys = Range.Double(0.0, 1.0, stepSize).map(s => (->(s), ->(sin(10 * s))))
    val zsys = Range.Double(0.0, 1.0, stepSize).map(s => (->(s), ->(cos(10 * s))))
    val f = Tanh
    val net = Network(Input(1) :: Hidden(10, f) :: Output(1, f) :: HNil,
      Settings(iterations = 2000, learningRate = 0.2, approximation = Some(Approximation(1E-9))))

    net.train(xsys.map(_._1), xsys.map(_._2))

    val res = net.evaluate(xsys.map(_._1))
    xsys.map(_._1).zip(res).foreach { case (l, r) => println(s"${l.head}, ${r.head}") }

  }

  def sequenceClassification = {

    import neuroflow.nets.LSTMNetwork._
    implicit val wp = RNN.WeightProvider(-0.2, 0.2)

    val stepSize = 0.1
    val xsys = Range.Double(0.0, 1.0, stepSize).map(s => (->(0.2), ->(-1.0)))
    val zsys = Range.Double(0.0, 1.0, stepSize).map(s => (->(0.8), ->(1.0)))
    val all = xsys ++ zsys
    val f = Tanh
    val net = Network(Input(1) :: Hidden(5, f) :: Output(1, f) :: HNil,
      Settings(iterations = 10000, learningRate = 0.2, approximation = Some(Approximation(1E-9))))

    net.train(all.map(_._1), all.map(_._2), Set(xsys.length))

    val resA = net.evaluateMean(xsys.map(_._1))
    val resB = net.evaluateMean(zsys.map(_._1))

    println(resA)
    println(resB)

  }

}
