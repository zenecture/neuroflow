package neuroflow.playground

import neuroflow.core.Activator._
import neuroflow.core.RNN.WeightProvider._
import neuroflow.core._
import neuroflow.nets.LSTMNetwork._
import neuroflow.application.plugin.Style._
import shapeless.HNil

/**
  * @author bogdanski
  * @since 07.07.16
  */

object Sequences {

  def apply = {

    val xsys = Range.Double(0.0, 1.0, 0.1).map(s => (->(s), ->(s + 0.1)))
    val net = Network(Input(1) :: Hidden(2, Tanh) :: Hidden(2, Tanh) :: Output(1, Tanh) :: HNil, Settings())

    net.evaluate(xsys.map(_._1))

  }

}
