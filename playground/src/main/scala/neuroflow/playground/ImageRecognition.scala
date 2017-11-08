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
import neuroflow.nets.gpu.ConvNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def apply = {

    val path = "/home/felix/cifar"
    val wps  = "/home/felix/cifar/waypoint"
    val lfo  = "/home/felix/cifar/lfo.txt"

    val classes =
      Seq("airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck")

    val classVecs = classes.zipWithIndex.map { case (c, i) => 
      c -> ~>(ζ[Float](classes.size)).io(_.update(i, 1.0f)).t
    }.toMap

    println("Loading data ...")

    val limits = (50000, 10000)

    val train = new File(path + "/train").list().take(limits._1).par.map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/train/" + s, None).map(_.mapValues(_.toFloat)) -> classVecs(c)
    }.seq

    val test = new File(path + "/test").list().take(limits._2).par.map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/test/" + s, None).map(_.mapValues(_.toFloat)) -> classVecs(c)
    }.seq

    classes.foreach { c =>
      println(s"|$c| = " + train.count(l => l._2 == classVecs(c)))
    }

    val f = ReLU

    val c1 = Convolution(dimIn = (32, 32, 3),  padding = 2`²`, field = 3`²`, stride = 1`²`, filters = 96, activator = f)
    val c2 = Convolution(dimIn = c1.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 96, activator = f)
    val c3 = Convolution(dimIn = c2.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 96, activator = f)
    val c4 = Convolution(dimIn = c3.dimOut,    padding = 1`²`, field = 3`²`, stride = 3`²`, filters = 192, activator = f)
    val c5 = Convolution(dimIn = c4.dimOut,    padding = 2`²`, field = 3`²`, stride = 1`²`, filters = 192, activator = f)
    val c6 = Convolution(dimIn = c5.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 192, activator = f)

    val convs = c1 :: c2 :: c3 :: c4 :: c5 :: c6 :: HNil

    val fullies =
      Dense(100, f)           ::
      Output(classes.size, f) :: HNil

    val config = (0 to 5).map(_ -> (0.001, 0.001)) :+ (6 -> (0.0001, 0.0001)) :+ (7 -> (0.01, 0.01))
    implicit val wp = neuroflow.core.WeightProvider.CNN[Float].normal(config.toMap)
//    implicit val wp = IO.File.readFloat(wps + "-iter-15.nf")

    val net = Network(convs ::: fullies,
      Settings[Float](
        prettyPrint     = true,
        learningRate    = { case (_, _) => 1E-5 },
        lossFunction    = Softmax(),
        updateRule      = Momentum(μ = 0.8f),
        iterations      = 20000,
        precision       = 1E-3,
        batchSize       = Some(32),
        lossFuncOutput  = Some(LossFuncOutput(Some(lfo))),
        waypoint        = Some(Waypoint(nth = 3, (iter, ws) => IO.File.write(ws, wps + s"-iter-$iter.nf")))
      )
    )

  net.train(train.map(_._1), train.map(_._2))

    val rate = test.map {
      case (x, y) =>
        val v = net(x)
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
