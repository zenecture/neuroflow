package neuroflow.nets.gpu

import breeze.linalg._
import breeze.stats._
import jcuda.jcublas.{JCublas2, cublasHandle}
import neuroflow.common.CanProduce
import neuroflow.core.Activator._
import neuroflow.core.IllusionBreaker.SettingsNotSupportedException
import neuroflow.core.Network._
import neuroflow.core._
import neuroflow.cuda._
import neuroflow.dsl._

import scala.annotation.tailrec
import scala.collection.Seq
import scala.collection.mutable.ArrayBuffer

/**
  *
  * Convolutional Neural Network running on CUDA,
  * uses gradient descent to optimize the loss function.
  *
  * @author bogdanski
  * @since 31.08.17
  *
  */

object ConvNetwork {

  implicit object double extends Constructor[Double, ConvNetworkDouble] {
    def apply(ls: Seq[Layer], loss: LossFunction[Double], settings: Settings[Double])(implicit breeder: WeightBreeder[Double]): ConvNetworkDouble = {
      ConvNetworkDouble(ls, loss, settings, breeder(ls))
    }
  }

  implicit object weights_double extends neuroflow.core.WeightBreeder.CNN_Builder[Double]


  implicit object single extends Constructor[Float, ConvNetworkFloat] {
    def apply(ls: Seq[Layer], loss: LossFunction[Float], settings: Settings[Float])(implicit breeder: WeightBreeder[Float]): ConvNetworkFloat = {
      ConvNetworkFloat(ls, loss, settings, breeder(ls))
    }
  }

  implicit object weights_float extends neuroflow.core.WeightBreeder.CNN_Builder[Float]

}



// <editor-fold defaultstate="collapsed" desc="Double Precision Impl">

case class ConvNetworkDouble(layers: Seq[Layer], lossFunction: LossFunction[Double], settings: Settings[Double], weights: Weights[Double],
                                           identifier: String = "neuroflow.nets.gpu.ConvNetwork", numericPrecision: String = "Double")
  extends CNN[Double] with WaypointLogic[Double] {

  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  type Vector   = DenseVector[Double]
  type Matrix   = DenseMatrix[Double]
  type Tensor   = Tensor3D[Double]
  type Vectors  = Seq[DenseVector[Double]]
  type Matrices = Seq[DenseMatrix[Double]]
  type Tensors  = Seq[Tensor3D[Double]]

  private val _allLayers  = layers.map {
    case d: Dense[Double]       => d
    case c: Convolution[Double] => c
  }.toArray

  private def activatorMapping(a: Activator[_], b: Double) = {
    a match {
      case x: ReLU[_]    => (CuMatrix.Activators.relu[Double]    , CuMatrix.Activators.relu_derivative[Double]    , b)
      case x: Linear[_]  => (CuMatrix.Activators.linear[Double]  , CuMatrix.Activators.linear_derivative[Double]  , b)
      case x: Sigmoid[_] => (CuMatrix.Activators.sigmoid[Double] , CuMatrix.Activators.sigmoid_derivative[Double] , b)
      case x: Tanh[_]    => (CuMatrix.Activators.tanh[Double]    , CuMatrix.Activators.tanh_derivative[Double]    , b)
      case x             => throw new SettingsNotSupportedException(s"This activator is not implemented for CUDA: ${a.symbol}.")
    }
  }
  
  private val _activators = _allLayers.map {
    case h: HasActivator[_] => h.activator match {
      case x: Activator[_] with Bias[Double] => activatorMapping(x.activator, x.bias)
      case x: Activator[_]                   => activatorMapping(x, 0.0)
    }
  }

  private val _lastLayerIdx = weights.size - 1

  private val _convLayers =
    _allLayers.zipWithIndex.map(_.swap).filter {
      case (_, _: Convolution[_]) => true
      case _                      => false
    }.toMap.mapValues {
      case c: Convolution[Double] => c
    }

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last

  private val _cuWeights = weights.map(m => CuMatrix.fromDense(m))


  /**
    * Computes output for `x`.
    */
  def apply(x: Tensor): Vector = {
    sink(x.matrix, _lastLayerIdx).toDenseVector
  }


  /**
    * `apply` under a focused layer.
    */
  def focus[L <: Layer](l: L)(implicit cp: CanProduce[(Matrix, L), l.algebraicType]): Tensor => l.algebraicType = {
    val lwi = layers.zipWithIndex
    val idx = lwi.find(_._1 eq l).orElse {
      val p = lwi.filter(_._1 == l)
      if (p.size > 1) warn(s"Focus layer $l is ambiguous. Taking first. " +
        "Alternatively, use a direct object reference to the desired layer.")
      p.headOption
    } match {
      case Some((l, i)) => debug(s"Found focus layer $l at index $i."); i
      case None => warn(s"Focus layer $l not found. Fallback to last layer."); _lastLayerIdx
    }
    (in: Tensor) => {
      cp(sink(in.matrix, idx), l)
    }
  }


  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Tensors, ys: Vectors): Unit = {
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    require(xs.size == ys.size, s"Mismatch between sample sizes. (${xs.size} != ${ys.size})")
    require(xs.size % batchSize == 0, s"Batches are not even. (${xs.size} % $batchSize = ${xs.size % batchSize} != 0)")
    if (settings.verbose) {
      info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt}.")
      info(s"Breeding batches ...")
    }
    val xsys = BatchBreeder.breedCNN(xs, ys, batchSize)
    gcThreshold match {
      case Some(bytes) => GcThreshold.set(bytes)
      case None        =>
    }
    run(xsys, learningRate(1 -> 1.0), batchSize, precision, batch = 0, batches = xsys.size, iteration = 1, iterations)
  }


  private def sink(x: Matrix, target: Int): Matrix = {
    val r1 = flow(x, target, batchSize = 1)
    val r2 = if (target == _lastLayerIdx) lossFunction.sink(r1) else r1
    r2
  }


  private def flow(in: Matrix, target: Int, batchSize: Int): Matrix = {

    val _fa = ArrayBuffer.empty[CuMatrix[Double]]
    val _fr = ArrayBuffer.empty[CuMatrix[Double]] // raw, unshaped

    @tailrec def conv(_in: CuMatrix[Double], i: Int): Unit = {
      val l = _convLayers(i)
      val p = _cuWeights(i) * convolute(_in, l, batchSize)
      p += _activators(i)._3
      val a = _activators(i)._1(p)
      _fa += { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      _fr += a
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: CuMatrix[Double], i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * _cuWeights(i)
      p += _activators(i)._3
      val a = _activators(i)._1(p)
      _fa += a
      _fr += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(CuMatrix.fromDense(in), 0)
    fully(_fa(_lastC), _lastC + 1)

    val r = _fr(target).toDense
    _fa.foreach(_.release())
    _fr.foreach(_.release())
    r

  }


  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[(Matrix, Matrix)], stepSize: Double, batchSize: Int, precision: Double,
                           batch: Int, batches: Int, iteration: Int, maxIterations: Int): Unit = {
    val (x, y) = (xsys(batch)._1, xsys(batch)._2)
    val loss =
      if (settings.approximation.isDefined) adaptWeightsApprox(x, y, stepSize, batchSize)
      else adaptWeights(x, y, stepSize, batchSize)
    val lossMean = mean(loss)
    if (settings.verbose) info(f"Iteration $iteration.${batch + 1}, Avg. Loss = $lossMean%.6g, Vector: $loss")
    maybeGraph(lossMean)
    waypoint(syncWeights)(iteration)
    if (lossMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize), batchSize,
        precision, (batch + 1) % batches, batches, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration of $maxIterations iterations.")
    }
  }


  /**
    * Copies batch to GPU, computes gradient for weights, updates weights using gradient descent and returns the loss matrix.
    */
  private def adaptWeights(x: Matrix, y: Matrix, stepSize: Double, batchSize: Int): Matrix = {

    import settings.updateRule

    val (_x, _y) = (CuMatrix.fromDense(x), CuMatrix.fromDense(y))

    val loss = CuMatrix.zeros[Double](batchSize, _outputDim)

    val fa  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
    val fb  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
    val fc  = collection.mutable.Map.empty[Int, CuMatrix[Double]]
    val dws = collection.mutable.Map.empty[Int, CuMatrix[Double]]
    val ds  = collection.mutable.Map.empty[Int, CuMatrix[Double]]

    @tailrec def conv(_in: CuMatrix[Double], i: Int): Unit = {
      val l = _convLayers(i)
      val c = convolute(_in, l, batchSize)
      val p = _cuWeights(i) * c
      p += _activators(i)._3
      val a = _activators(i)._1(p)
      val b = _activators(i)._2(p)
      fa += i -> {
        if (i == _lastC) {
          val rb = reshape_batch(a, l.dimOut, batchSize)
          a.release()
          rb
        } else a
      }
      fb += i -> b
      fc += i -> c
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: CuMatrix[Double], i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * _cuWeights(i)
      p += _activators(i)._3
      val a = _activators(i)._1(p)
      val b = _activators(i)._2(p)
      fa += i -> a
      fb += i -> b
      if (i < _lastL) fully(a, i + 1)
    }

    @tailrec def derive(i: Int): Unit = {
      if (i == _lastLayerIdx) {
        val (err, grad) = lossFunction(_y, fa(i))
        val d = grad *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        loss += err
        derive(i - 1)
      } else if (i < _lastLayerIdx && i > _lastC) {
        val d = (ds(i + 1) * _cuWeights(i + 1).t) *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        derive(i - 1)
      } else if (i == _lastC) {
        val l = _convLayers(i)
        val d1 = ds(i + 1) * _cuWeights(i + 1).t
        val d2 = reshape_batch_backprop(d1, l.dimOut, batchSize)
        val d = d2 *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      } else {
        val l = _convLayers(i + 1)
        val ww = reshape_batch(_cuWeights(i + 1), (l.field._1, l.field._2, l.filters), l.dimIn._3)
        val dc = convolute_backprop(ds(i + 1), l, batchSize)
        val d = ww * dc *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      }
    }

    conv(_x, 0)
    fully(fa(_lastC), _lastC + 1)
    derive(_lastLayerIdx)

    ds.values.foreach(_.release())
    fa.values.foreach(_.release())
    fb.values.foreach(_.release())
    fc.values.foreach(_.release())

    (0 to _lastLayerIdx).foreach(i => updateRule(_cuWeights(i), dws(i), stepSize, i))

    dws.values.foreach(_.release())
    _x.release()
    _y.release()

    val lossReduced = (loss.t * CuMatrix.ones[Double](loss.rows, 1)).t
    lossReduced.toDense

  }


  /** For debugging, approximates the gradients using `settings.approximation`. */
  private def adaptWeightsApprox(xs: Matrix, ys: Matrix, stepSize: Double, batchSize: Int): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable[Double]])
    val _rule: Debuggable[Double] = settings.updateRule.asInstanceOf[Debuggable[Double]]

    def lossFunc(): Matrix = {
      val loss = lossFunction(ys, flow(xs, _lastLayerIdx, batchSize))._1
      val reduced = (loss.t * DenseMatrix.ones[Double](loss.rows, 1)).t
      reduced
    }

    val out = lossFunc()

    def approximateGradient(weightLayer: Int, weight: (Int, Int)): Double = {
      sum(settings.approximation.get.apply(weights, lossFunc, syncWithGPU, weightLayer, weight))
    }

    def syncWithGPU(): Unit = {
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
          val grad = approximateGradient(idx, k)
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

    syncWithGPU()

    out

  }

  private def syncWeights(): Unit = {
    weights.zip(_cuWeights).foreach {
      case (w, cw) => w := cw.toDense
    }
  }

}

// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="Single Precision Impl">

case class ConvNetworkFloat(layers: Seq[Layer], lossFunction: LossFunction[Float], settings: Settings[Float], weights: Weights[Float],
                                          identifier: String = "neuroflow.nets.gpu.ConvNetwork", numericPrecision: String = "Single")
  extends CNN[Float] with WaypointLogic[Float] {

  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  type Vector   = DenseVector[Float]
  type Matrix   = DenseMatrix[Float]
  type Tensor   = Tensor3D[Float]
  type Vectors  = Seq[DenseVector[Float]]
  type Matrices = Seq[DenseMatrix[Float]]
  type Tensors  = Seq[Tensor3D[Float]]

  private val _allLayers  = layers.map {
    case d: Dense[Float]        => d
    case c: Convolution[Float]  => c
  }.toArray

  private def activatorMapping(a: Activator[_], b: Float) = {
    a match {
      case x: ReLU[_]    => (CuMatrix.Activators.relu[Float]    , CuMatrix.Activators.relu_derivative[Float]    , b)
      case x: Linear[_]  => (CuMatrix.Activators.linear[Float]  , CuMatrix.Activators.linear_derivative[Float]  , b)
      case x: Sigmoid[_] => (CuMatrix.Activators.sigmoid[Float] , CuMatrix.Activators.sigmoid_derivative[Float] , b)
      case x: Tanh[_]    => (CuMatrix.Activators.tanh[Float]    , CuMatrix.Activators.tanh_derivative[Float]    , b)
      case x             => throw new SettingsNotSupportedException(s"This activator is not implemented for CUDA: ${a.symbol}.")
    }
  }

  private val _activators = _allLayers.map {
    case h: HasActivator[_] => h.activator match {
      case x: Activator[_] with Bias[Float]  => activatorMapping(x.activator, x.bias)
      case x: Activator[_]                   => activatorMapping(x, 0.0f)
    }
  }

  private val _lastLayerIdx = weights.size - 1

  private val _convLayers =
    _allLayers.zipWithIndex.map(_.swap).filter {
      case (_, _: Convolution[_]) => true
      case _                      => false
    }.toMap.mapValues {
      case c: Convolution[Float]  => c
    }

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last

  private val _cuWeights = weights.map(m => CuMatrix.fromDense(m))

  /**
    * Computes output for `x`.
    */
  def apply(x: Tensor): Vector = {
    sink(x.matrix, _lastLayerIdx).toDenseVector
  }


  /**
    * `apply` under a focused layer.
    */
  def focus[L <: Layer](l: L)(implicit cp: CanProduce[(Matrix, L), l.algebraicType]): Tensor => l.algebraicType = {
    val lwi = layers.zipWithIndex
    val idx = lwi.find(_._1 eq l).orElse {
      val p = lwi.filter(_._1 == l)
      if (p.size > 1) warn(s"Focus layer $l is ambiguous. Taking first. " +
        "Alternatively, use a direct object reference to the desired layer.")
      p.headOption
    } match {
      case Some((l, i)) => debug(s"Found focus layer $l at index $i."); i
      case None => warn(s"Focus layer $l not found. Fallback to last layer."); _lastLayerIdx
    }
    (in: Tensor) => {
      cp(sink(in.matrix, idx), l)
    }
  }


  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Tensors, ys: Vectors): Unit = {
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    require(xs.size == ys.size, s"Mismatch between sample sizes. (${xs.size} != ${ys.size})")
    require(xs.size % batchSize == 0, s"Batches are not even. (${xs.size} % $batchSize = ${xs.size % batchSize} != 0)")
    if (settings.verbose) {
      info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toFloat / batchSize.toFloat).toInt}.")
      info(s"Breeding batches ...")
    }
    val xsys = BatchBreeder.breedCNN(xs, ys, batchSize)
    gcThreshold match {
      case Some(bytes) => GcThreshold.set(bytes)
      case None        =>
    }
    run(xsys, learningRate(1 -> 1.0).toFloat, batchSize, precision, batch = 0, batches = xsys.size, iteration = 1, iterations)
  }


  private def sink(x: Matrix, target: Int): Matrix = {
    val r1 = flow(x, target, batchSize = 1)
    val r2 = if (target == _lastLayerIdx) lossFunction.sink(r1) else r1
    r2
  }


  private def flow(in: Matrix, target: Int, batchSize: Int): Matrix = {

    val _fa = ArrayBuffer.empty[CuMatrix[Float]]
    val _fr = ArrayBuffer.empty[CuMatrix[Float]] // raw, unshaped

    @tailrec def conv(_in: CuMatrix[Float], i: Int): Unit = {
      val l = _convLayers(i)
      val p = _cuWeights(i) * convolute(_in, l, batchSize)
      p += _activators(i)._3
      val a = _activators(i)._1(p)
      _fa += { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      _fr += a
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: CuMatrix[Float], i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * _cuWeights(i)
      p += _activators(i)._3
      val a = _activators(i)._1(p)
      _fa += a
      _fr += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(CuMatrix.fromDense(in), 0)
    fully(_fa(_lastC), _lastC + 1)

    val r = _fr(target).toDense
    _fa.foreach(_.release())
    _fr.foreach(_.release())
    r

  }


  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[(Matrix, Matrix)], stepSize: Float, batchSize: Int, precision: Double,
                           batch: Int, batches: Int, iteration: Int, maxIterations: Int): Unit = {
    val (x, y) = (xsys(batch)._1, xsys(batch)._2)
    val loss =
      if (settings.approximation.isDefined) adaptWeightsApprox(x, y, stepSize, batchSize)
      else adaptWeights(x, y, stepSize, batchSize)
    val lossMean = mean(loss)
    if (settings.verbose) info(f"Iteration $iteration.${batch + 1}, Avg. Loss = $lossMean%.6g, Vector: $loss")
    maybeGraph(lossMean)
    waypoint(syncWeights)(iteration)
    if (lossMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize).toFloat, batchSize,
        precision, (batch + 1) % batches, batches, iteration + 1, maxIterations)
    } else {
      info(f"Took $iteration of $maxIterations iterations.")
    }
  }


  /**
    * Copies batch to GPU, computes gradient for weights, updates weights using gradient descent and returns the loss matrix.
    */
  private def adaptWeights(x: Matrix, y: Matrix, stepSize: Float, batchSize: Int): Matrix = {

    import settings.updateRule

    val (_x, _y) = (CuMatrix.fromDense(x), CuMatrix.fromDense(y))

    val loss = CuMatrix.zeros[Float](batchSize, _outputDim)

    val fa  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
    val fb  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
    val fc  = collection.mutable.Map.empty[Int, CuMatrix[Float]]
    val dws = collection.mutable.Map.empty[Int, CuMatrix[Float]]
    val ds  = collection.mutable.Map.empty[Int, CuMatrix[Float]]

    @tailrec def conv(_in: CuMatrix[Float], i: Int): Unit = {
      val l = _convLayers(i)
      val c = convolute(_in, l, batchSize)
      val p = _cuWeights(i) * c
      p += _activators(i)._3
      val a = _activators(i)._1(p)
      val b = _activators(i)._2(p)
      fa += i -> {
        if (i == _lastC) {
          val rb = reshape_batch(a, l.dimOut, batchSize)
          a.release()
          rb
        } else a
      }
      fb += i -> b
      fc += i -> c
      p.release()
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: CuMatrix[Float], i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * _cuWeights(i)
      p += _activators(i)._3
      val a = _activators(i)._1(p)
      val b = _activators(i)._2(p)
      fa += i -> a
      fb += i -> b
      p.release()
      if (i < _lastL) fully(a, i + 1)
    }

    @tailrec def derive(i: Int): Unit = {
      if (i == _lastLayerIdx) {
        val (err, grad) = lossFunction(_y, fa(i))
        val d = grad *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        loss += err
        err.release()
        grad.release()
        derive(i - 1)
      } else if (i < _lastLayerIdx && i > _lastC) {
        val d = (ds(i + 1) * _cuWeights(i + 1).t) *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        derive(i - 1)
      } else if (i == _lastC) {
        val l = _convLayers(i)
        val d1 = ds(i + 1) * _cuWeights(i + 1).t
        val d2 = reshape_batch_backprop(d1, l.dimOut, batchSize)
        val d = d2 *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        d1.release()
        d2.release()
        if (i > 0) derive(i - 1)
      } else {
        val l = _convLayers(i + 1)
        val ww = reshape_batch(_cuWeights(i + 1), (l.field._1, l.field._2, l.filters), l.dimIn._3)
        val dc = convolute_backprop(ds(i + 1), l, batchSize)
        val d = ww * dc *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        ww.release()
        dc.release()
        if (i > 0) derive(i - 1)
      }
    }

    conv(_x, 0)
    fully(fa(_lastC), _lastC + 1)
    derive(_lastLayerIdx)

    ds.values.foreach(_.release())
    fa.values.foreach(_.release())
    fb.values.foreach(_.release())
    fc.values.foreach(_.release())

    (0 to _lastLayerIdx).foreach(i => updateRule(_cuWeights(i), dws(i), stepSize, i))

    dws.values.foreach(_.release())
    _x.release()
    _y.release()

    val lossReduced = (loss.t * CuMatrix.ones[Float](loss.rows, 1)).t
    lossReduced.toDense

  }


  /** For debugging, approximates the gradients using `settings.approximation`. */
  private def adaptWeightsApprox(xs: Matrix, ys: Matrix, stepSize: Float, batchSize: Int): Matrix = {

    require(settings.updateRule.isInstanceOf[Debuggable[Float]])
    val _rule: Debuggable[Float] = settings.updateRule.asInstanceOf[Debuggable[Float]]

    def lossFunc(): Matrix = {
      val loss = lossFunction(ys, flow(xs, _lastLayerIdx, batchSize))._1
      val reduced = (loss.t * DenseMatrix.ones[Float](loss.rows, 1)).t
      reduced
    }

    val out = lossFunc()

    def approximateGradient(weightLayer: Int, weight: (Int, Int)): Float = {
      sum(settings.approximation.get.apply(weights, lossFunc, syncWithGPU, weightLayer, weight))
    }

    def syncWithGPU(): Unit = {
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
          val grad = approximateGradient(idx, k)
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

    syncWithGPU()

    out

  }

  private def syncWeights(): Unit = {
    weights.zip(_cuWeights).foreach {
      case (w, cw) => w := cw.toDense
    }
  }

}

// </editor-fold>

