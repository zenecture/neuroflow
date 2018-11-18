package neuroflow.nets.cpu

import breeze.linalg._
import breeze.stats._
import neuroflow.core.Network._
import neuroflow.core.WaypointLogic.NoOp
import neuroflow.core.{CanProduce, _}
import neuroflow.dsl._

import scala.annotation.tailrec
import scala.collection.Seq
import scala.collection.mutable.ArrayBuffer
import scala.util.Try

/**
  *
  * Convolutional Neural Network running on CPU,
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

  implicit object weights_double extends neuroflow.core.WeightBreeder.Initializer[Double]


  implicit object single extends Constructor[Float, ConvNetworkFloat] {
    def apply(ls: Seq[Layer], loss: LossFunction[Float], settings: Settings[Float])(implicit breeder: WeightBreeder[Float]): ConvNetworkFloat = {
      ConvNetworkFloat(ls, loss, settings, breeder(ls))
    }
  }

  implicit object weights_float extends neuroflow.core.WeightBreeder.Initializer[Float]

}

//<editor-fold defaultstate="collapsed" desc="Double Precision Impl">

case class ConvNetworkDouble(layers: Seq[Layer], lossFunction: LossFunction[Double], settings: Settings[Double], weights: Weights[Double],
                                           identifier: String = "neuroflow.nets.cpu.ConvNetwork", numericPrecision: String = "Double")
  extends CNN[Double] with WaypointLogic[Double] {

  type Vector   = DenseVector[Double]
  type Matrix   = DenseMatrix[Double]
  type Tensor   = Tensor3D[Double]
  type Vectors  = Seq[DenseVector[Double]]
  type Matrices = Seq[DenseMatrix[Double]]
  type Tensors  = Seq[Tensor3D[Double]]

  private val _allLayers = layers.map {
    case d: Dense[Double]         => d
    case c: Convolution[Double]   => c
  }.toArray

  private val _lastLayerIdx = weights.size - 1

  private val _convLayers =
    _allLayers.zipWithIndex.map(_.swap).filter {
      case (_, _: Convolution[_]) => true
      case _                      => false
    }.toMap.mapValues {
      case c: Convolution[Double] => c
    }

  private val _activators = _allLayers.map(_.activator)

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last


  /**
    * Computes output for `x`.
    */
  def apply(x: Tensor): Vector = {
    sink(x.matrix, _lastLayerIdx, batchSize = 1).toDenseVector
  }


  /**
    * Computes output for given inputs `in`
    * using efficient batch mode.
    */
  def batchApply(xs: Tensors): Vectors = {
    BatchBreeder.unsliceMatrixByRow {
      sink(BatchBreeder.horzCatTensorBatch(xs), _lastLayerIdx, batchSize = xs.size)
    }
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
      cp(sink(in.matrix, idx, batchSize = 1), l)
    }
  }


  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Tensors, ys: Vectors): Try[Run] = Try {
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    require(xs.size == ys.size, s"Mismatch between sample sizes. (${xs.size} != ${ys.size})")
    if (settings.verbose) {
      if(xs.size % batchSize != 0) warn(s"Batches are not even. (${xs.size} % $batchSize = ${xs.size % batchSize} != 0)")
      info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt}.")
      info(s"Breeding batches ...")
    }
    val (xsys, batchSizes) = BatchBreeder.breedCNN(xs, ys, batchSize)
    run(xsys, learningRate(1 -> 1.0), batchSizes, precision, batch = 0, batches = xsys.size, iteration = 1, iterations, startTime = System.currentTimeMillis())
  }


  private def sink(x: Matrix, target: Int, batchSize: Int): Matrix = {
    val r1 = flow(x, target, batchSize)
    val r2 = if (target == _lastLayerIdx) lossFunction.sink(r1) else r1
    r2
  }


  private def flow(in: Matrix, target: Int, batchSize: Int): Matrix = {

    val _fa = ArrayBuffer.empty[Matrix]
    val _fr = ArrayBuffer.empty[Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val p = weights(i) * convolute(_in, l, batchSize)
      val a = p.map(_activators(i))
      _fa += { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      _fr += a
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(_activators(i))
      _fa += a
      _fr += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(in, 0)
    fully(_fa(_lastC), _lastC + 1)

    _fr(target)

  }


  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[(Matrix, Matrix)], stepSize: Double, batchSizes: Map[Int, Int], precision: Double,
                           batch: Int, batches: Int, iteration: Int, maxIterations: Int, startTime: Long): Run = {
    val batchSize = batchSizes(batch)
    val (x, y) = (xsys(batch)._1, xsys(batch)._2)
    val loss =
      if (settings.approximation.isDefined) adaptWeightsApprox(x, y, stepSize, batchSize)
      else adaptWeights(x, y, stepSize, batchSize)
    val lossMean = mean(loss)
    if (settings.verbose) info(f"Iteration $iteration.${batch + 1}, Avg. Loss = $lossMean%.6g, Vector: $loss")
    maybeGraph(lossMean)
    waypoint(NoOp)(iteration)
    if (lossMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize), batchSizes,
        precision, (batch + 1) % batches, batches, iteration + 1, maxIterations, startTime)
    } else {
      info(f"Took $iteration of $maxIterations iterations.")
      Run(startTime, System.currentTimeMillis(), iteration)
    }
  }


  /**
    * Computes gradient for weights with respect to given batch,
    * adapts their value using gradient descent and returns the loss matrix.
    */
  private def adaptWeights(x: Matrix, y: Matrix, stepSize: Double, batchSize: Int): Matrix = {

    import settings.updateRule

    val loss = DenseMatrix.zeros[Double](batchSize, _outputDim)

    val fa  = collection.mutable.Map.empty[Int, Matrix]
    val fb  = collection.mutable.Map.empty[Int, Matrix]
    val fc  = collection.mutable.Map.empty[Int, Matrix]
    val dws = collection.mutable.Map.empty[Int, Matrix]
    val ds  = collection.mutable.Map.empty[Int, Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val c = convolute(_in, l, batchSize)
      val p = weights(i) * c
      val a = p.map(_activators(i))
      val b = p.map(_activators(i).derivative)
      fa += i -> { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      fb += i -> b
      fc += i -> c
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(_activators(i))
      val b = p.map(_activators(i).derivative)
      fa += i -> a
      fb += i -> b
      if (i < _lastL) fully(a, i + 1)
    }

    @tailrec def derive(i: Int): Unit = {
      if (i == _lastLayerIdx) {
        val (err, grad) = lossFunction(y, fa(i))
        val d = grad *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        loss += err
        derive(i - 1)
      } else if (i < _lastLayerIdx && i > _lastC) {
        val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        derive(i - 1)
      } else if (i == _lastC) {
        val l = _convLayers(i)
        val d1 = ds(i + 1) * weights(i + 1).t
        val d2 = reshape_batch_backprop(d1, l.dimOut, batchSize)
        val d = d2 *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      } else {
        val l = _convLayers(i + 1)
        val ww = reshape_batch(weights(i + 1), (l.field._1, l.field._2, l.filters), l.dimIn._3)
        val dc = convolute_backprop(ds(i + 1), l, batchSize)
        val d = ww * dc *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      }
    }

    conv(x, 0)
    fully(fa(_lastC), _lastC + 1)
    derive(_lastLayerIdx)

    (0 to _lastLayerIdx).foreach(i => updateRule(weights(i), dws(i), stepSize, i))

    val lossReduced = (loss.t * DenseMatrix.ones[Double](loss.rows, 1)).t
    lossReduced

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
      sum(settings.approximation.get.apply(weights, lossFunc, () => (), weightLayer, weight))
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

    out

  }

}

//</editor-fold>

//<editor-fold defaultstate="collapsed" desc="Single Precision Impl">

case class ConvNetworkFloat(layers: Seq[Layer], lossFunction: LossFunction[Float], settings: Settings[Float], weights: Weights[Float],
                                          identifier: String = "neuroflow.nets.cpu.ConvNetwork", numericPrecision: String = "Single")
  extends CNN[Float] with WaypointLogic[Float] {

  type Vector   = DenseVector[Float]
  type Matrix   = DenseMatrix[Float]
  type Tensor   = Tensor3D[Float]
  type Vectors  = Seq[DenseVector[Float]]
  type Matrices = Seq[DenseMatrix[Float]]
  type Tensors  = Seq[Tensor3D[Float]]

  private val _allLayers = layers.map {
    case d: Dense[Float]         => d
    case c: Convolution[Float]   => c
  }.toArray

  private val _lastLayerIdx = weights.size - 1

  private val _convLayers =
    _allLayers.zipWithIndex.map(_.swap).filter {
      case (_, _: Convolution[_]) => true
      case _                      => false
    }.toMap.mapValues {
      case c: Convolution[Float]  => c
    }

  private val _activators = _allLayers.map(_.activator)

  private val _outputDim = _allLayers.last.neurons
  private val _lastC     = _convLayers.maxBy(_._1)._1
  private val _lastL     = _allLayers.indices.last


  /**
    * Computes output for `x`.
    */
  def apply(x: Tensor): Vector = {
    sink(x.matrix, _lastLayerIdx, batchSize = 1).toDenseVector
  }


  /**
    * Computes output for given inputs `in`
    * using efficient batch mode.
    */
  def batchApply(xs: Tensors): Vectors = {
    BatchBreeder.unsliceMatrixByRow {
      sink(BatchBreeder.horzCatTensorBatch(xs), _lastLayerIdx, batchSize = xs.size)
    }
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
      cp(sink(in.matrix, idx, batchSize = 1), l)
    }
  }


  /**
    * Trains this net with input `xs` against output `ys`.
    */
  def train(xs: Tensors, ys: Vectors): Try[Run] = Try {
    import settings._
    val batchSize = settings.batchSize.getOrElse(xs.size)
    require(xs.size == ys.size, s"Mismatch between sample sizes. (${xs.size} != ${ys.size})")
    if (settings.verbose) {
      if(xs.size % batchSize != 0) warn(s"Batches are not even. (${xs.size} % $batchSize = ${xs.size % batchSize} != 0)")
      info(s"Training with ${xs.size} samples, batch size = $batchSize, batches = ${math.ceil(xs.size.toDouble / batchSize.toDouble).toInt}.")
      info(s"Breeding batches ...")
    }
    val (xsys, batchSizes) = BatchBreeder.breedCNN(xs, ys, batchSize)
    run(xsys, learningRate(1 -> 1.0f), batchSizes, precision, batch = 0, batches = xsys.size, iteration = 1, iterations, startTime = System.currentTimeMillis())
  }


  private def sink(x: Matrix, target: Int, batchSize: Int): Matrix = {
    val r1 = flow(x, target, batchSize)
    val r2 = if (target == _lastLayerIdx) lossFunction.sink(r1) else r1
    r2
  }


  private def flow(in: Matrix, target: Int, batchSize: Int): Matrix = {

    val _fa = ArrayBuffer.empty[Matrix]
    val _fr = ArrayBuffer.empty[Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val p = weights(i) * convolute(_in, l, batchSize)
      val a = p.map(_activators(i))
      _fa += { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      _fr += a
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(_activators(i))
      _fa += a
      _fr += a
      if (i < _lastL) fully(a, i + 1)
    }

    conv(in, 0)
    fully(_fa(_lastC), _lastC + 1)

    _fr(target)

  }


  /**
    * The training loop.
    */
  @tailrec private def run(xsys: Seq[(Matrix, Matrix)], stepSize: Float, batchSizes: Map[Int, Int], precision: Double,
                           batch: Int, batches: Int, iteration: Int, maxIterations: Int, startTime: Long): Run = {
    val batchSize = batchSizes(batch)
    val (x, y) = (xsys(batch)._1, xsys(batch)._2)
    val loss =
      if (settings.approximation.isDefined) adaptWeightsApprox(x, y, stepSize, batchSize)
      else adaptWeights(x, y, stepSize, batchSize)
    val lossMean = mean(loss)
    if (settings.verbose) info(f"Iteration $iteration.${batch + 1}, Avg. Loss = $lossMean%.6g, Vector: $loss")
    maybeGraph(lossMean)
    waypoint(NoOp)(iteration)
    if (lossMean > precision && iteration < maxIterations) {
      run(xsys, settings.learningRate(iteration + 1 -> stepSize), batchSizes,
        precision, (batch + 1) % batches, batches, iteration + 1, maxIterations, startTime)
    } else {
      info(f"Took $iteration of $maxIterations iterations.")
      Run(startTime, System.currentTimeMillis(), iteration)
    }
  }


  /**
    * Computes gradient for weights with respect to given batch,
    * adapts their value using gradient descent and returns the loss matrix.
    */
  private def adaptWeights(x: Matrix, y: Matrix, stepSize: Float, batchSize: Int): Matrix = {

    import settings.updateRule

    val loss = DenseMatrix.zeros[Float](batchSize, _outputDim)

    val fa  = collection.mutable.Map.empty[Int, Matrix]
    val fb  = collection.mutable.Map.empty[Int, Matrix]
    val fc  = collection.mutable.Map.empty[Int, Matrix]
    val dws = collection.mutable.Map.empty[Int, Matrix]
    val ds  = collection.mutable.Map.empty[Int, Matrix]

    @tailrec def conv(_in: Matrix, i: Int): Unit = {
      val l = _convLayers(i)
      val c = convolute(_in, l, batchSize)
      val p = weights(i) * c
      val a = p.map(_activators(i))
      val b = p.map(_activators(i).derivative)
      fa += i -> { if (i == _lastC) reshape_batch(a, l.dimOut, batchSize) else a }
      fb += i -> b
      fc += i -> c
      if (i < _lastC) conv(a, i + 1)
    }

    @tailrec def fully(_in: Matrix, i: Int): Unit = {
      val l = _allLayers(i)
      val p = _in * weights(i)
      val a = p.map(_activators(i))
      val b = p.map(_activators(i).derivative)
      fa += i -> a
      fb += i -> b
      if (i < _lastL) fully(a, i + 1)
    }

    @tailrec def derive(i: Int): Unit = {
      if (i == _lastLayerIdx) {
        val (err, grad) = lossFunction(y, fa(i))
        val d = grad *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        loss += err
        derive(i - 1)
      } else if (i < _lastLayerIdx && i > _lastC) {
        val d = (ds(i + 1) * weights(i + 1).t) *:* fb(i)
        val dw = fa(i - 1).t * d
        dws += i -> dw
        ds += i -> d
        derive(i - 1)
      } else if (i == _lastC) {
        val l = _convLayers(i)
        val d1 = ds(i + 1) * weights(i + 1).t
        val d2 = reshape_batch_backprop(d1, l.dimOut, batchSize)
        val d = d2 *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      } else {
        val l = _convLayers(i + 1)
        val ww = reshape_batch(weights(i + 1), (l.field._1, l.field._2, l.filters), l.dimIn._3)
        val dc = convolute_backprop(ds(i + 1), l, batchSize)
        val d = ww * dc *:* fb(i)
        val dw = d * fc(i).t
        dws += i -> dw
        ds += i -> d
        if (i > 0) derive(i - 1)
      }
    }

    conv(x, 0)
    fully(fa(_lastC), _lastC + 1)
    derive(_lastLayerIdx)

    (0 to _lastLayerIdx).foreach(i => updateRule(weights(i), dws(i), stepSize, i))

    val lossReduced = (loss.t * DenseMatrix.ones[Float](loss.rows, 1)).t
    lossReduced

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
      sum(settings.approximation.get.apply(weights, lossFunc, () => (), weightLayer, weight))
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

    out

  }

}

//</editor-fold>