package neuroflow.playground

import breeze.linalg.max
import neuroflow.application.plugin.IO._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core.Convolution.IntTupler
import neuroflow.core._
import neuroflow.nets.cpu.ConvNetwork._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def apply = {

    val path = "/Users/felix/github/unversioned/cifar"
    val wps  = path + "/waypoint"
    val lfo  = path + "/lfo.txt"

    val classes =
      Seq("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    val classVecs = classes.zipWithIndex.map { case (c, i) =>
      c -> ~>(ζ[Double](classes.size)).io(_.update(i, 1.0)).t
    }.toMap

    println("Loading data ...")

    val limits = (32, 10)

    val train = new java.io.File(path + "/train").list().take(limits._1).par.map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/train/" + s) -> classVecs(c)
    }.seq

    val test = new java.io.File(path + "/test").list().take(limits._2).par.map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/test/" + s) -> classVecs(c)
    }.seq

    classes.foreach { c =>
      println(s"|$c| = " + train.count(l => l._2 == classVecs(c)))
    }

    val f = ReLU

    val c1 = Convolution(dimIn = (32, 32, 3),  padding = 2`²`, field = 3`²`, stride = 1`²`, filters = 16, activator = f)
    val c2 = Convolution(dimIn = c1.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 16, activator = f)
    val c3 = Convolution(dimIn = c2.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 16, activator = f)
    val c4 = Convolution(dimIn = c3.dimOut,    padding = 1`²`, field = 3`²`, stride = 3`²`, filters = 32, activator = f)
    val c5 = Convolution(dimIn = c4.dimOut,    padding = 2`²`, field = 3`²`, stride = 1`²`, filters = 32, activator = f)
    val c6 = Convolution(dimIn = c5.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 32, activator = f)

    val L = c1 :: c2 :: c3 :: c4 :: c5 :: c6 :: Dense(100, f) :: Softmax()

    val config = (0 to 5).map(_ -> (0.01, 0.01)) :+ (6 -> (0.001, 0.001)) :+ (7 -> (0.01, 0.01))
    implicit val wp = neuroflow.core.WeightProvider.CNN[Double].normal(config.toMap)
//    implicit val wp = IO.File.readDouble(wps + "-iter-2460.nf")

    val net = Network(
      layout = L,
      Settings[Double](
        prettyPrint     = true,
        learningRate    = { case (_, _) => 1E-5 },
        updateRule      = Momentum(μ = 0.8),
        iterations      = 20000,
        precision       = 1E-3,
        batchSize       = Some(32),
        lossFuncOutput  = Some(LossFuncOutput(Some(lfo))),
        waypoint        = Some(Waypoint(nth = 30, (iter, ws) => File.write(ws, wps + s"-iter-$iter.nf")))
      )
    )

  net.train(train.map(_._1), train.map(_._2))

    val rate = test.map {
      case (x, y) =>
        val v = net(x)
        println(v)
        val c = v.toArray.indexOf(max(v))
        val t = y.toArray.indexOf(max(y))
        if (c == t) 1.0 else 0.0
    }.sum / test.size.toDouble

    println(s"Recognition rate = ${rate * 100.0} %, Error rate = ${(1.0 - rate) * 100.0} %!")

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"Pos: $posWeights, Neg: $negWeights")

  }

}
