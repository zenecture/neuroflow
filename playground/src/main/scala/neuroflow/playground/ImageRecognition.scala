package neuroflow.playground

import breeze.linalg.max
import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO
import neuroflow.application.plugin.IO._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.dsl.Convolution.IntTupler
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.nets.cpu.ConvNetwork._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def apply = {

    val path = "/Users/felix/github/unversioned/cifar"
//    val path = "/home/felix"
    val wps  = path + "/waypoint"
    val lfo  = path + "/lfo.txt"

    val classes =
      Seq("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    val classVecs = classes.zipWithIndex.map { case (c, i) =>
      c -> ~>(ζ[Float](classes.size)).io(_.update(i, 1.0f)).t
    }.toMap

    println("Loading data ...")

    val limits = (256, 32)

    val train = new java.io.File(path + "/train").list().take(limits._1).par.map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/train/" + s).float -> classVecs(c)
    }.seq

    val test = new java.io.File(path + "/test").list().take(limits._2).par.map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/test/" + s).float -> classVecs(c)
    }.seq

    classes.foreach { c =>
      println(s"|$c| = " + train.count(l => l._2 == classVecs(c)))
    }

    val f = ReLU

    val c1 = Convolution(dimIn = (32, 32, 3),  padding = 2`²`, field = 3`²`, stride = 1`²`, filters = 48, activator = f)
    val c2 = Convolution(dimIn = c1.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 48, activator = f)
    val c3 = Convolution(dimIn = c2.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 48, activator = f)
    val c4 = Convolution(dimIn = c3.dimOut,    padding = 1`²`, field = 3`²`, stride = 3`²`, filters = 96, activator = f)
    val c5 = Convolution(dimIn = c4.dimOut,    padding = 2`²`, field = 3`²`, stride = 1`²`, filters = 96, activator = f)
    val c6 = Convolution(dimIn = c5.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 96, activator = f)

    val L = c1 :: c2 :: c3 :: c4 :: c5 :: c6 :: Dense(100, f) :: Dense(10, f) :: Softmax()

    implicit val wp = neuroflow.core.WeightProvider[Float].normal(Map(
      0 -> (0.01, 0.01), 1 -> (0.01, 0.01), 2 -> (0.01, 0.01),
      3 -> (0.001, 0.001), 4 -> (0.001, 0.001), 5 -> (0.001, 0.001),
      6 -> (0.0001, 0.0001), 7 -> (0.01, 0.01)
    ))

//    implicit val wp = IO.File.readWeights[Float](wps + "-iter-1000.nf")

    val net = Network(
      layout = L,
      Settings[Float](
        prettyPrint     = true,
        learningRate    = { case _ => 1E-7 },
        updateRule      = Momentum(μ = 0.8f),
        iterations      = 100000,
        precision       = 1E-3,
        batchSize       = Some(256),
        gcThreshold     = Some(100 * 1024 * 1024L),
        lossFuncOutput  = Some(LossFuncOutput(Some(lfo))),
        waypoint        = Some(Waypoint(nth = 1000, (iter, ws) => File.writeWeights(ws, wps + s"-iter-$iter.nf")))
      )
    )

  net.train(train.map(_._1), train.map(_._2))

    val rate = train.map {
      case (x, y) =>
        val v = net(x)
//        println(v)
        val c = v.toArray.indexOf(max(v))
        val t = y.toArray.indexOf(max(y))
        if (c == t) 1.0 else 0.0
    }.sum / train.size.toDouble

    println(s"Recognition rate = ${rate * 100.0} %, Error rate = ${(1.0 - rate) * 100.0} %!")

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"Pos: $posWeights, Neg: $negWeights")

  }

}
