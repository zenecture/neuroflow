package neuroflow.nets.gpu

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import jcuda.jcublas.{JCublas2, cublasHandle}
import neuroflow.core.Activator._
import neuroflow.core.EarlyStoppingLogic.CanAverage
import neuroflow.core.IllusionBreaker.SettingsNotSupportedException
import neuroflow.core.Network._
import neuroflow.core._
import neuroflow.cuda._

import scala.annotation.tailrec
import scala.collection.Seq


/**
  *
  * This is a feed-forward neural network with fully connected layers running on CUDA.
  * It uses gradient descent to optimize the specified loss function.
  *
  * @author bogdanski
  * @since 15.01.16
  *
  */


object DenseNetwork {
  
  implicit val double: Constructor[Double, DenseNetworkDouble] = new Constructor[Double, DenseNetworkDouble] {
    def apply(ls: Seq[Layer], settings: Settings[Double])(implicit weightProvider: WeightProvider[Double]): DenseNetworkDouble = {
      DenseNetworkDouble(ls, settings, weightProvider(ls))
    }
  }

  implicit val single: Constructor[Float, DenseNetworkSingle] = new Constructor[Float, DenseNetworkSingle] {
    def apply(ls: Seq[Layer], settings: Settings[Float])(implicit weightProvider: WeightProvider[Float]): DenseNetworkSingle = {
      DenseNetworkSingle(ls, settings, weightProvider(ls))
    }
  }
  
}


//<editor-fold defaultstate="collapsed" desc="Double Precision Impl">

private[nets] case class DenseNetworkDouble(layers: Seq[Layer], settings: Settings[Double], weights: Weights[Double],
                                            identifier: String = "neuroflow.nets.gpu.DenseNetwork", numericPrecision: String = "Double")
  extends FFN[Double] with EarlyStoppingLogic[Double] with KeepBestLogic[Double] with WaypointLogic[Double] {

  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  type Vector   = Network.Vector[Double]
  type Vectors  = Network.Vectors[Double]
  type Matrix   = Network.Matrix[Double]
  type Matrices = Network.Matrices[Double]

  private val _layers = layers.map {
    case Focus(inner) => inner
    case layer: Layer => layer
  }.toArray

  private val _focusLayer     = layers.collect { case c: Focus[_] => c }.headOption

  private val _activators     = _layers.tail.map {
    case h: HasActivator[_] => h.activator match {
      case ReLU    => CuMatrix.Activators.relu[Double]    -> CuMatrix.Activators.relu_derivative[Double]
      case Linear  => CuMatrix.Activators.linear[Double]  -> CuMatrix.Activators.linear_derivative[Double]
      case Sigmoid => CuMatrix.Activators.sigmoid[Double] -> CuMatrix.Activators.sigmoid_derivative[Double]
      case Tanh    => CuMatrix.Activators.tanh[Double]    -> CuMatrix.Activators.tanh_derivative[Double]
      case x       => throw new SettingsNotSupportedException(s"This activator is not implemented for CUDA: $x.")
    }
  }

  private val _outputDim      = _layers.last.neurons
  private val _lastWlayerIdx  = weights.size - 1
  private val _cuWeights      = weights.map(m => CuMatrix.fromDense(m))
  
  private implicit object Average extends CanAverage[Double, DenseNetworkDouble, Vector, Vector] {
    def averagedError(xs: Vectors, ys: Vectors): Double = {
      val errors = xs.map(evaluate).zip(ys).map {
        case (a, b) => mean(abs(a - b))
      }
      mean(errors)
    }
  }

  /**
    * Checks if the [[Settings]] are properly defined.
    * Might throw a [[SettingsNotSupportedException]].
    */
  override def checkSettings(): Unit = {
    super.checkSettings()
    if (settings.specifics.isDefined)
      warn("No specific settings supported. This has no effect.")
    settings.regularization.foreach {
      case _: EarlyStopping[_, _] | KeepBest =>
      case _ => throw new SettingsNotSupportedException("This regularization is not supported.")
    }
  }

  /**
    * Computes output for `x`.
    */
  def apply(x: Vector): Vector = {
    val input = DenseMatrix.create[Double](1, x.size, x.toArray)
    _focusLayer.map { cl =>
      val i = layers.indexOf(cl) - 1
      val r = flow(input, i)
      r
    }.getOrElse {
      val r = flow(input, _lastWlayerIdx)
      settings.lossFunction match {
        case _: SquaredMeanError[_] => r
        case _: Softmax[_]          => SoftmaxImpl(r)
        case _                      => r
      }
    }.toDenseVector
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors, ys: Vectors): Unit = {
    require(xs.size == ys.size, "Mismatch between sample sizes!")
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    if (settings.verbose) info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt} ...")
    val xsys = xs.map(_.asDenseMatrix).zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq.map { xy =>
      xy.map(_._1).reduce(DenseMatrix.vertcat(_, _)) -> xy.map(_._2).reduce(DenseMatrix.vertcat(_, _))
    }
    GcThreshold.set(this, batchSize * 2)
    run(xsys, learningRate(1 -> 1.0), precision, batch = 0, batches = xsys.size, iteration = 1, iterations)
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[(Matrix, Matrix)], stepSize: Double, precision: Double, batch: Int,
                           batches: Int, iteration: Int, maxIterations: Int): Unit = {
    val (x, y) = (xsys(batch)._1, xsys(batch)._2)
    val loss =
      if (settings.approximation.isDefined) adaptWeightsApprox(x, y, stepSize)
      else adaptWeights(x, y, stepSize)
    val lossMean = mean(loss)
    if (settings.verbose) info(f"Iteration $iteration. Ø Batch Loss: $lossMean%.6g. Loss Vector: $loss")
    syncWeights()
    maybeGraph(lossMean)
    waypoint(iteration)
    if (lossMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize), precision, (batch + 1) % batches, batches, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration iterations of $maxIterations with Loss = $lossMean%.6g")
    }
  }

  /**
    * Computes the network recursively.
    */
  private def flow(in: Matrix, outLayer: Int): Matrix = {
    val _in = CuMatrix.fromDense(in)
    val fa  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
    @tailrec def forward(in: CuMatrix[Double], i: Int): Unit = {
      val p = in * _cuWeights(i)
      val a = _activators(i)._1(p)
      p.release()
      fa += i -> a
      if (i < outLayer) forward(a, i + 1)
    }
    forward(_in, 0)
    val o = fa(outLayer).toDense
    _in.release()
    fa.values.foreach(_.release())
    o
  }

  /**
    * Copies batch to GPU, then computes gradient for all weights,
    * adapts their value using gradient descent and returns the error matrix.
    */
  private def adaptWeights(xs: Matrix, ys: Matrix, stepSize: Double): Matrix = {

    import settings.lossFunction

    val x = CuMatrix.fromDense(xs)
    val y = CuMatrix.fromDense(ys)

    val errSum = CuMatrix.zeros[Double](xs.rows, _outputDim)

    val fa  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
    val fb  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
    val dws = collection.mutable.Map.empty[Int, CuMatrix[Double]]
    val ds  = collection.mutable.Map.empty[Int, CuMatrix[Double]]

    @tailrec def forward(in: CuMatrix[Double], i: Int): Unit = {
      val p = in * _cuWeights(i)
      val a = _activators(i)._1(p)
      val b = _activators(i)._2(p)
      p.release()
      fa += i -> a
      fb += i -> b
      if (i < _lastWlayerIdx) forward(a, i + 1)
    }

    @tailrec def derive(i: Int): Unit = {
      if (i == 0 && _lastWlayerIdx == 0) {
        val (err, grad) = lossFunction(y, fa(i))
        val d = grad *:* fb(i)
        val dw = x.t * d
        dws += i -> dw
        errSum += err
        d.release()
        err.release()
        grad.release()
      } else if (i == _lastWlayerIdx) {
        val (err, grad) = lossFunction(y, fa(i))
        val d = grad *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        errSum += err
        err.release()
        grad.release()
        derive(i - 1)
      } else if (i < _lastWlayerIdx && i > 0) {
        val d1 = ds(i + 1) * _cuWeights(i + 1).t
        val d2 = d1 *:* fb(i)
        val dw = fa(i - 1).t * d2
        dws += i -> dw
        ds += i -> d2
        d1.release()
        derive(i - 1)
      } else if (i == 0) {
        val d1 = ds(i + 1) * _cuWeights(i + 1).t
        val d2 = d1 *:* fb(i)
        val dw = x.t * d2
        dws += i -> dw
        d1.release()
      }
    }

    forward(x, 0)
    derive(_lastWlayerIdx)

    ds.values.foreach(_.release())
    fa.values.foreach(_.release())
    fb.values.foreach(_.release())

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(_cuWeights(i), dws(i), stepSize, i)
      i += 1
    }

    x.release()
    y.release()

    dws.values.foreach(_.release())

    val errSumReduced = (errSum.toDense.t * DenseMatrix.ones[Double](errSum.rows, 1)).t
    errSum.release()

    errSumReduced

  }

  private def syncWeights(): Unit = {
    weights.zip(_cuWeights).foreach {
      case (w, cw) => w := cw.toDense
    }
  }

  /** Approximates the gradient based on finite central differences. (For debugging) */
  private def adaptWeightsApprox(xs: Matrix, ys: Matrix, stepSize: Double): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable[Double]])
    val _rule: Debuggable[Double] = settings.updateRule.asInstanceOf[Debuggable[Double]]

    def errorFunc(): Matrix = {
      val errSum = settings.lossFunction(ys, flow(xs, _lastWlayerIdx))._1
      val errSumReduced = (errSum.t * DenseMatrix.ones[Double](errSum.rows, 1)).t
      errSumReduced
    }

    val out = errorFunc()

    def approximateErrorFuncDerivative(weightLayer: Int, weight: (Int, Int)): Matrix = {
      val Δ = settings.approximation.get.Δ
      val v = weights(weightLayer)(weight)
      weights(weightLayer).update(weight, v - Δ)
      syncWeightsBack()
      val a = errorFunc()
      weights(weightLayer).update(weight, v + Δ)
      syncWeightsBack()
      val b = errorFunc()
      weights(weightLayer).update(weight, v)
      syncWeightsBack()
      (b - a) / (2 * Δ)
    }

    def syncWeightsBack(): Unit = {
      weights.zip(_cuWeights).foreach {
        case (w, cw) => cw := w
      }
    }

    val updates = collection.mutable.HashMap.empty[(Int, (Int, Int)), Double]
    val grads   = collection.mutable.HashMap.empty[(Int, (Int, Int)), Double]
    val debug   = collection.mutable.HashMap.empty[Int, Matrix]

    weights.zipWithIndex.foreach {
      case (l, idx) =>
        debug += idx -> l.copy
        l.foreachPair { (k, v) =>
          val efd  = approximateErrorFuncDerivative(idx, k)
          val grad = sum(efd)
          updates += (idx, k) -> (v - (stepSize * grad))
          grads += (idx, k) -> grad
        }
    }

    updates.foreach {
      case ((wl, k), v) =>
        weights(wl).update(k, v)
    }

    grads.foreach {
      case ((wl, k), v) =>
        debug(wl).update(k, v)
    }

    _rule.lastGradients = debug

    syncWeightsBack()

    out

  }

}


//</editor-fold>


//<editor-fold defaultstate="collapsed" desc="Single Precision Impl">

private[nets] case class DenseNetworkSingle(layers: Seq[Layer], settings: Settings[Float], weights: Weights[Float],
                                            identifier: String = "neuroflow.nets.gpu.DenseNetwork", numericPrecision: String = "Single")
  extends FFN[Float] with EarlyStoppingLogic[Float] with KeepBestLogic[Float] with WaypointLogic[Float] {

  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  type Vector   = Network.Vector[Float]
  type Vectors  = Network.Vectors[Float]
  type Matrix   = Network.Matrix[Float]
  type Matrices = Network.Matrices[Float]

  private val _layers = layers.map {
    case Focus(inner) => inner
    case layer: Layer => layer
  }.toArray

  private val _focusLayer     = layers.collect { case c: Focus[_] => c }.headOption

  private val _activators     = _layers.tail.map {
    case h: HasActivator[_] => h.activator match {
      case ReLU    => CuMatrix.Activators.relu[Float]    -> CuMatrix.Activators.relu_derivative[Float]
      case Linear  => CuMatrix.Activators.linear[Float]  -> CuMatrix.Activators.linear_derivative[Float]
      case Sigmoid => CuMatrix.Activators.sigmoid[Float] -> CuMatrix.Activators.sigmoid_derivative[Float]
      case Tanh    => CuMatrix.Activators.tanh[Float]    -> CuMatrix.Activators.tanh_derivative[Float]
      case x       => throw new SettingsNotSupportedException(s"This activator is not implemented for CUDA: $x.")
    }
  }

  private val _outputDim      = _layers.last.neurons
  private val _lastWlayerIdx  = weights.size - 1
  private val _cuWeights      = weights.map(m => CuMatrix.fromDense(m))

  private implicit object Average extends CanAverage[Float, DenseNetworkSingle, Vector, Vector] {
    def averagedError(xs: Vectors, ys: Vectors): Double = {
      val errors = xs.map(evaluate).zip(ys).map {
        case (a, b) => mean(abs(a - b))
      }
      mean(errors)
    }
  }

  /**
    * Checks if the [[Settings]] are properly defined.
    * Might throw a [[SettingsNotSupportedException]].
    */
  override def checkSettings(): Unit = {
    super.checkSettings()
    if (settings.specifics.isDefined)
      warn("No specific settings supported. This has no effect.")
    settings.regularization.foreach {
      case _: EarlyStopping[_, _] | KeepBest =>
      case _ => throw new SettingsNotSupportedException("This regularization is not supported.")
    }
  }

  /**
    * Computes output for `x`.
    */
  def apply(x: Vector): Vector = {
    val input = DenseMatrix.create[Float](1, x.size, x.toArray)
    _focusLayer.map { cl =>
      val i = layers.indexOf(cl) - 1
      val r = flow(input, i)
      r
    }.getOrElse {
      val r = flow(input, _lastWlayerIdx)
      settings.lossFunction match {
        case _: SquaredMeanError[_] => r
        case _: Softmax[_]          => SoftmaxImpl(r)
        case _                      => r
      }
    }.toDenseVector
  }

  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Vectors, ys: Vectors): Unit = {
    require(xs.size == ys.size, "Mismatch between sample sizes!")
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    if (settings.verbose) info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt} ...")
    val xsys = xs.map(_.asDenseMatrix).zip(ys.map(_.asDenseMatrix)).grouped(batchSize).toSeq.map { xy =>
      xy.map(_._1).reduce(DenseMatrix.vertcat(_, _)) -> xy.map(_._2).reduce(DenseMatrix.vertcat(_, _))
    }
    GcThreshold.set(this, batchSize * 2)
    run(xsys, learningRate(1 -> 1.0).toFloat, precision, batch = 0, batches = xsys.size, iteration = 1, iterations)
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[(Matrix, Matrix)], stepSize: Float, precision: Double, batch: Int,
                           batches: Int, iteration: Int, maxIterations: Int): Unit = {
    val (x, y) = (xsys(batch)._1, xsys(batch)._2)
    val loss =
      if (settings.approximation.isDefined) adaptWeightsApprox(x, y, stepSize)
      else adaptWeights(x, y, stepSize)
    val lossMean = mean(loss)
    if (settings.verbose) info(f"Iteration $iteration. Ø Batch Loss: $lossMean%.6g. Loss Vector: $loss")
    syncWeights()
    maybeGraph(lossMean)
    waypoint(iteration)
    if (lossMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize).toFloat, precision, (batch + 1) % batches, batches, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration iterations of $maxIterations with Loss = $lossMean%.6g")
    }
  }

  /**
    * Computes the network recursively.
    */
  private def flow(in: Matrix, outLayer: Int): Matrix = {
    val _in = CuMatrix.fromDense(in)
    val fa  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
    @tailrec def forward(in: CuMatrix[Float], i: Int): Unit = {
      val p = in * _cuWeights(i)
      val a = _activators(i)._1(p)
      p.release()
      fa += i -> a
      if (i < outLayer) forward(a, i + 1)
    }
    forward(_in, 0)
    val o = fa(outLayer).toDense
    _in.release()
    fa.values.foreach(_.release())
    o
  }

  /**
    * Copies batch to GPU, then computes gradient for all weights,
    * adapts their value using gradient descent and returns the error matrix.
    */
  private def adaptWeights(xs: Matrix, ys: Matrix, stepSize: Float): Matrix = {

    import settings.lossFunction

    val x = CuMatrix.fromDense(xs)
    val y = CuMatrix.fromDense(ys)

    val errSum = CuMatrix.zeros[Float](xs.rows, _outputDim)

    val fa  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
    val fb  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
    val dws = collection.mutable.Map.empty[Int, CuMatrix[Float]]
    val ds  = collection.mutable.Map.empty[Int, CuMatrix[Float]]

    @tailrec def forward(in: CuMatrix[Float], i: Int): Unit = {
      val p = in * _cuWeights(i)
      val a = _activators(i)._1(p)
      val b = _activators(i)._2(p)
      p.release()
      fa += i -> a
      fb += i -> b
      if (i < _lastWlayerIdx) forward(a, i + 1)
    }

    @tailrec def derive(i: Int): Unit = {
      if (i == 0 && _lastWlayerIdx == 0) {
        val (err, grad) = lossFunction(y, fa(i))
        val d = grad *:* fb(i)
        val dw = x.t * d
        dws += i -> dw
        errSum += err
        d.release()
        err.release()
        grad.release()
      } else if (i == _lastWlayerIdx) {
        val (err, grad) = lossFunction(y, fa(i))
        val d = grad *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        errSum += err
        err.release()
        grad.release()
        derive(i - 1)
      } else if (i < _lastWlayerIdx && i > 0) {
        val d1 = ds(i + 1) * _cuWeights(i + 1).t
        val d2 = d1 *:* fb(i)
        val dw = fa(i - 1).t * d2
        dws += i -> dw
        ds += i -> d2
        d1.release()
        derive(i - 1)
      } else if (i == 0) {
        val d1 = ds(i + 1) * _cuWeights(i + 1).t
        val d2 = d1 *:* fb(i)
        val dw = x.t * d2
        dws += i -> dw
        d1.release()
      }
    }

    forward(x, 0)
    derive(_lastWlayerIdx)

    ds.values.foreach(_.release())
    fa.values.foreach(_.release())
    fb.values.foreach(_.release())

    var i = 0
    while (i <= _lastWlayerIdx) {
      settings.updateRule(_cuWeights(i), dws(i), stepSize, i)
      i += 1
    }

    x.release()
    y.release()

    dws.values.foreach(_.release())

    val errSumReduced = (errSum.toDense.t * DenseMatrix.ones[Float](errSum.rows, 1)).t
    errSum.release()

    errSumReduced

  }

  private def syncWeights(): Unit = {
    weights.zip(_cuWeights).foreach {
      case (w, cw) => w := cw.toDense
    }
  }

  /** Approximates the gradient based on finite central differences. (For debugging) */
  private def adaptWeightsApprox(xs: Matrix, ys: Matrix, stepSize: Float): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable[Float]])
    val _rule: Debuggable[Float] = settings.updateRule.asInstanceOf[Debuggable[Float]]

    def errorFunc(): Matrix = {
      val errSum = settings.lossFunction(ys, flow(xs, _lastWlayerIdx))._1
      val errSumReduced = (errSum.t * DenseMatrix.ones[Float](errSum.rows, 1)).t
      errSumReduced
    }

    val out = errorFunc()

    def approximateErrorFuncDerivative(weightLayer: Int, weight: (Int, Int)): Matrix = {
      val Δ = settings.approximation.get.Δ.toFloat
      val v = weights(weightLayer)(weight)
      weights(weightLayer).update(weight, v - Δ)
      syncWeightsBack()
      val a = errorFunc()
      weights(weightLayer).update(weight, v + Δ)
      syncWeightsBack()
      val b = errorFunc()
      weights(weightLayer).update(weight, v)
      syncWeightsBack()
      (b - a) / (2 * Δ)
    }

    def syncWeightsBack(): Unit = {
      weights.zip(_cuWeights).foreach {
        case (w, cw) => cw := w
      }
    }

    val updates = collection.mutable.HashMap.empty[(Int, (Int, Int)), Float]
    val grads   = collection.mutable.HashMap.empty[(Int, (Int, Int)), Float]
    val debug   = collection.mutable.HashMap.empty[Int, Matrix]

    weights.zipWithIndex.foreach {
      case (l, idx) =>
        debug += idx -> l.copy
        l.foreachPair { (k, v) =>
          val efd  = approximateErrorFuncDerivative(idx, k)
          val grad = sum(efd)
          updates += (idx, k) -> (v - (stepSize * grad))
          grads += (idx, k) -> grad
        }
    }

    updates.foreach {
      case ((wl, k), v) =>
        weights(wl).update(k, v)
    }

    grads.foreach {
      case ((wl, k), v) =>
        debug(wl).update(k, v)
    }

    _rule.lastGradients = debug

    syncWeightsBack()

    out

  }

}

//</editor-fold>
