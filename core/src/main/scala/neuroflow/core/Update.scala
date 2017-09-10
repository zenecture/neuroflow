package neuroflow.core

import Network._

/**
  * @author bogdanski
  * @since 09.09.17
  */


trait Update extends ((Matrix, Matrix, Double, Int) => Unit)

case object Vanilla extends Update {
  def apply(ws: Matrix, dws: Matrix, learningRate: Double, position: Int): Unit = {
    ws -= (dws *= learningRate)
  }
}

case class Momentum(μ: Double = 0.9) extends Update {
  private val vs = collection.mutable.HashMap.empty[Int, Matrix]
  def apply(ws: Matrix, dws: Matrix, learningRate: Double, position: Int): Unit = {
    if (vs.isDefinedAt(position)) vs(position) := (μ * vs(position)) - (dws *= learningRate)
    else vs += position -> -(dws * learningRate)
    ws += vs(position)
  }
}

case class Debuggable() extends Update {
  var lastGradients = collection.mutable.Map.empty[Int, Matrix]
  def apply(ws: Matrix, dws: Matrix, learningRate: Double, position: Int): Unit = {
    lastGradients += position -> dws
    ws -= (dws *= learningRate)
  }
}
