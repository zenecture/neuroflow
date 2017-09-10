package neuroflow.nets.distributed

import java.util.UUID

import akka.actor.{Actor, ActorRef, ActorSelection, ActorSystem, PoisonPill, Props}
import akka.pattern.ask
import akka.util.Timeout
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import com.typesafe.config.ConfigFactory
import neuroflow.common.Logs
import neuroflow.core.EarlyStoppingLogic.CanAverage
import neuroflow.core.IllusionBreaker.SettingsNotSupportedException
import neuroflow.core.Network._
import neuroflow.core._
import neuroflow.nets.Registry

import scala.annotation.tailrec
import scala.collection.Seq
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}


/**
  *
  * This is a feed-forward neural network with fully connected layers.
  * It uses distributed dynamic gradient descent to optimize the error function Σ1/2(y - net(x))².
  *
  * Use the parallelism parameter with care, as it greatly affects memory usage.
  *
  * @author bogdanski
  * @since 15.01.16
  *
  */


object DefaultNetwork {
  implicit val constructor: Constructor[DefaultNetwork] = new Constructor[DefaultNetwork] {
    def apply(ls: Seq[Layer], settings: Settings)(implicit weightProvider: WeightProvider): DefaultNetwork = {
      DefaultNetwork(ls, settings, weightProvider(ls))
    }
  }
}


private[nets] case class DefaultNetwork(layers: Seq[Layer], settings: Settings, weights: Weights,
                                        identifier: String = Registry.register())
  extends DistributedFeedForwardNetwork with EarlyStoppingLogic with KeepBestLogic {

  import neuroflow.core.Network._

  private val _layers = layers.map {
    case Focus(inner) => inner
    case layer: Layer   => layer
  }.toArray

  private val _clusterLayer   = layers.collect { case c: Focus => c }.headOption

  private val _lastWlayerIdx  = weights.size - 1
  private def _weightsWi      = weights.map(_.data.zipWithIndex.grouped(settings.transport.messageGroupSize)).zipWithIndex
  private val _weightsRoCo    = weights.map(w => w.rows -> w.cols)
  private val _layersNI       = _layers.tail.map { case h: HasActivator[Double] => h }
  private val _outputDim      = _layers.last.neurons

  private val _akka = ActorSystem("NeuroFlow", ConfigFactory.parseString(
    s"""
      |akka {
      |  log-dead-letters = 0
      |  extensions = ["com.romix.akka.serialization.kryo.KryoSerializationExtension$$"]
      |  actor {
      |    provider = remote
      |    kryo {
      |      type = "nograph"
      |      idstrategy = "incremental"
      |      implicit-registration-logging = true
      |    }
      |    serializers {
      |      kryo = "com.twitter.chill.akka.AkkaSerializer"
      |    }
      |    serialization-bindings {
      |       "neuroflow.nets.distributed.Message" = kryo
      |    }
      |  }
      |  remote {
      |    artery {
      |      enabled = on
      |      canonical.hostname = "${settings.coordinator.host}"
      |      canonical.port = ${settings.coordinator.port}
      |      advanced {
      |        maximum-frame-size = ${settings.transport.frameSize}
      |        maximum-large-frame-size = ${settings.transport.frameSize}
      |      }
      |    }
      |  }
      |}
    """.stripMargin))

  private implicit object Average extends CanAverage[DefaultNetwork, Vector, Vector] {
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
    if (settings.approximation.isDefined)
      throw new SettingsNotSupportedException("Doesn't work in distributed mode.")
    settings.regularization.foreach {
      case _: EarlyStopping[_, _] | KeepBest =>
      case _ => throw new SettingsNotSupportedException("This regularization is not supported.")
    }
  }

  /**
    * Triggers execution of training for nodes `ns`.
    */
  def train(ns: collection.Set[Node]): Unit = {
    import settings._
    implicit val to = Timeout(100 days)
    val xs = ns.map(n => _akka.actorSelection(s"akka://NeuroFlow@${n.host}:${n.port}/user/executor")).toSeq
    xs.foreach { x =>
      Await.result(x ? Heartbeat, atMost = 120 seconds)
      if (verbose) info(s"Connected to $x.")
    }
    run(xs, learningRate(0), precision, 1, iterations)
  }


  /**
    * Computes output for `x`.
    */
  def apply(x: Vector): Vector = {
    val input = DenseMatrix.create[Double](1, x.size, x.toArray)
    _clusterLayer.map { cl =>
      flow(input, layers.indexOf(cl) - 1).toDenseVector
    }.getOrElse {
      flow(input, _lastWlayerIdx).toDenseVector
    }
  }

  /**
    * The training loop.
    */
  @tailrec private def run(xs: Seq[ActorSelection], stepSize: Double, precision: Double, iteration: Int, maxIterations: Int): Unit = {
    val error = adaptWeights(xs, stepSize)
    val errorMean = mean(error)
    if (settings.verbose) info(f"Iteration $iteration - Mean Error $errorMean%.6g - Error Vector $error")
    maybeGraph(errorMean)
    keepBest(errorMean, weights)
    if (errorMean > precision && iteration < maxIterations && !shouldStopEarly) {
      run(xs, settings.learningRate(iteration + 1), precision, iteration + 1, maxIterations)
    } else {
      if (settings.verbose) info(f"Took $iteration iterations of $maxIterations with Mean Error = $errorMean%.6g")
      takeBest()
    }
  }

  /**
    * Computes the network recursively.
    */
  private def flow(in: Matrix, outLayer: Int): Matrix = {
    val fa  = collection.mutable.Map.empty[Int, Matrix]
    @tailrec def forward(in: Matrix, i: Int): Unit = {
      val p = in * weights(i)
      val a = p.map(_layersNI(i).activator)
      fa += i -> a
      if (i < outLayer) forward(a, i + 1)
    }
    forward(in, 0)
    fa(outLayer)
  }

  /**
    * Asks all nodes for derivatives and returns error matrix.
    */
  private def adaptWeights(xs: Seq[ActorSelection], stepSize: Double): Matrix = {

    import _akka.dispatcher
    implicit val to = Timeout(100 days)

    val processors = xs.map { x =>
      _akka.actorOf(Props(new ProcessorActor(x, stepSize, _weightsRoCo, _outputDim, _weightsWi, layers, settings)))
    }

    val results = Await.result(Future.sequence {
      processors.map { processor =>
        (processor ? 'Execute).mapTo[(Weights, Matrix)]
      }
    }, atMost = Duration.Inf)

    val (newWeights, error) = results.foldLeft(results.head) {
      case ((w1, e1), (w2, e2)) =>
        w1.zip(w2).foreach { case (a, b) => a += b }
        (w1, e1 += e2)
    }

    weights.zip(newWeights).foreach {
      case (o, n) => o -= n
    }

    error

  }

}

class ProcessorActor(x: ActorSelection, stepSize: Double, _weightsRoCo: IndexedSeq[(Int, Int)],
                     _outputDim: Int, _weightsWi: IndexedSeq[(Iterator[Array[(Double, Int)]], Int)],
                     layers: Seq[Layer], settings: Settings) extends Actor with Logs {

  import context.dispatcher

  private val _weights = _weightsRoCo.map {
    case (rows, cols) => DenseMatrix.create[Double](rows, cols, Array.fill(rows * cols)(0.0))
  }
  private val _error   = DenseMatrix.zeros[Double](1, _outputDim)
  private val _id      = UUID.randomUUID.toString
  private val _batches = _weightsWi.flatMap {
    case (wi, j) => wi.map { d => WeightBatch(UUID.randomUUID.toString, _id, d, j) }
  }

  private val _scheduler  = context.system.scheduler

  private val _acks       = collection.mutable.HashMap.empty[String, Message]

  private val _batchC_T   = _batches.length
  private val _batchE_T   = _error.data.zipWithIndex.grouped(settings.transport.messageGroupSize).size
  private var _batchC     = 0
  private var _batchE     = 0

  private var _request: ActorRef = _

  def receive: Receive = {
    case 'Execute =>
      _request = sender()
      _scheduler.schedule(1 minute, 10 seconds, self, 'Resend)
      x ! Job(_id, _batches.length, layers, _weightsRoCo, stepSize, settings.parallelism)
      _batches.foreach { wb =>
        _acks += wb.id -> wb
        x ! wb
      }

    case 'Resend =>
      debug(s"Resending ${_acks.size} batches ...")
      _acks.values.foreach(x ! _)

    case Ack(id) =>
      debug(s"Got ack $id")
      _acks -= id

    case b @ WeightBatch(id, jobId, _, position) if position < 0 =>
      debug(s"Got malformed WeightBatch $b. Discarding it ...")

    case WeightBatch(id, jobId, data, position) =>
      debug(s"Got WeightBatch for jobId = $jobId")
      sender ! Ack(id)
      _batchC += 1
      data.foreach { case (v, i) => _weights(position).data.update(i, v) }
      maybeComplete()

    case ErrorBatch(id, jobId, data) =>
      debug(s"Got ErrorBatch for jobId = $jobId")
      sender ! Ack(id)
      _batchE += 1
      data.foreach { case (v, i) => _error.data.update(i, v) }
      maybeComplete()

  }

  private def maybeComplete(): Unit =
    if ((_batchC >= _batchC_T) && (_batchE >= _batchE_T)) {
    _request ! (_weights, _error)
    self ! PoisonPill
  }

}
