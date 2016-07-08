package neuroflow.playground

import neuroflow.core.Activator._
import neuroflow.core._
import neuroflow.nets.LSTMNetwork._
import neuroflow.application.plugin.Style._
import shapeless.HNil

/**
  * @author bogdanski
  * @since 07.07.16
  */

object Sequences {

  implicit val wp = RNN.WeightProvider(-0.2, 0.2)

  def apply = {

    val xsys = Range.Double(0.0, 1.0, 0.1).map(s => (->(s), ->(if (s < 0.5) 0.0 else 0.5)))
    val net = Network(Input(1) :: Hidden(2, Sigmoid) :: Output(1, Sigmoid) :: HNil, Settings(iterations = 1000, approximation = Some(Approximation(1E-5))))
    net.train(xsys.map(_._1), xsys.map(_._2))
    val test = net.evaluate(xsys.map(_._1))

    xsys.map(_._1).zip(test).foreach { case (l, r) => println(s"${l.head}, ${r.head}") }
    println("No. of weights: " + net.weights.map(_.size).sum)
    println("Weights: " + net.weights)

  }

}
