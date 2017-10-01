package neuroflow.playground


import java.io.File

import breeze.linalg.max
import neuroflow.application.plugin.IO
import neuroflow.application.plugin.IO._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core.Convolution.IntTupler
import neuroflow.core._
import neuroflow.nets.cpu.ConvNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def apply = {

    val path = "/Users/felix/github/unversioned/cifar"
    val wps  = "/Users/felix/github/unversioned/cifarWP"
    val efo  = "/Users/felix/github/unversioned/efo.txt"

    implicit val wp = neuroflow.core.WeightProvider.Float.CNN.normal(0.0f, 0.01f)
//    implicit val wp = IO.File.readFloat(wps)

    val classes =
      Seq("airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck")

    val classVecs = classes.zipWithIndex.map { case (c, i) => 
      c -> ~>(ζ[Float](classes.size)).io(_.update(i, 1.0f)).t
    }.toMap

    val train = new File(path + "/train").list().take(50000).map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/train/" + s, None).map(_.mapValues(_.toFloat)) -> classVecs(c)
    }

    val test = new File(path + "/test").list().take(10000).map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/test/" + s, None).map(_.mapValues(_.toFloat)) -> classVecs(c)
    }

    classes.foreach { c =>
      println(s"|$c| = " + train.count(l => l._2 == classVecs(c)))
    }

    val f = ReLU

    val a = Convolution(dimIn = (32, 32, 3), padding = 2`²`, field = 3`²`, stride = 1`²`, filters = 96, activator = f)
    val b = Convolution(dimIn = a.dimOut,    padding = 2`²`, field = 3`²`, stride = 1`²`, filters = 96, activator = f)
    val c = Convolution(dimIn = b.dimOut,    padding = 2`²`, field = 3`²`, stride = 1`²`, filters = 96, activator = f)

    val convs   = a :: b :: c :: HNil

    val fullies =
      Dense(100, f)           ::
      Output(classes.size, f) :: HNil

    val net = Network(convs ::: fullies,
      Settings[Float](
        prettyPrint     = true,
        learningRate    = { case (_, _) => 1E-4 },
        updateRule      = Momentum(μ = 0.9f),
        iterations      = 20000,
        batchSize       = Some(8),
        parallelism     = Some(8),
        errorFuncOutput = Some(ErrorFuncOutput(Some(efo))),
        waypoint        = Some(Waypoint(nth = 3, (iter, ws) => IO.File.write(ws, wps + s"$iter.nf")))
      )
    )

    net.train(train.map(_._1), train.map(_._2))

    val rate = train.map {
      case (x, y) =>
        val v = net(x)
        val c = v.toArray.indexOf(max(v))
        val t = y.toArray.indexOf(max(y))
        println(s"${classes(t)} classified as ${classes(c)}")
        println(net(x))
        if (c == t) 1.0 else 0.0
    }.sum / train.size.toDouble

    println(s"Recognition rate = ${rate * 100.0} %, Error rate = ${(1.0 - rate) * 100.0} %!")

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"Pos: $posWeights, Neg: $negWeights")

  }

}
