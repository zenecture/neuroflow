package neuroflow.playground


import java.io.File

import breeze.linalg.max
import neuroflow.application.plugin.IO
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.common.~>
import neuroflow.core.Activator._
import neuroflow.core.Convolution.IntTupler
import neuroflow.core._
import neuroflow.nets.ConvNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def apply = {

    val path = "/Users/felix/github/unversioned/cifar"
    val wps  = "/Users/felix/github/unversioned/cifarWP.nf"
    val efo  = "/Users/felix/github/unversioned/efo.txt"

    implicit val wp = neuroflow.core.CNN.WeightProvider(0.001, 0.01)
//    implicit val wp = IO.File.read(wps)

    val classes =
      Seq("airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck")

    val classVecs = classes.zipWithIndex.map { case (c, i) => c -> ~>(ζ(classes.size)).io(_.update(i, 1.0)).t }.toMap

    val train = new File(path + "/train").list().take(1000).map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/train/" + s, None) -> classVecs(c)
    }

    val test = new File(path + "/test").list().take(10).map { s =>
      val c = classes.find(z => s.contains(z)).get
      extractRgb3d(path + "/test/" + s, None) -> classVecs(c)
    }

    classes.foreach { c =>
      println(s"|$c| = " + train.count(l => l._2 == classVecs(c)))
    }

    val f = ReLU

    val a = Convolution((32, 32, 3), field = 3`²`, filters = 96, stride = 1, f)
    val b = Convolution( a.dimOut,   field = 4`²`, filters = 64, stride = 2, f)
    val c = Convolution( b.dimOut,   field = 4`²`, filters = 32, stride = 2, f)

    val convs = a :: b :: c :: HNil
    val fully = Dense(20, f) :: Output(classes.size, f) :: HNil

    val net = Network(convs ::: fully,
      Settings(
        prettyPrint = true,
        learningRate = { case (_, _) => 1E-5 },
        updateRule = Vanilla,
        iterations = 1000,
        parallelism = 8,
        batchSize = Some(8),
        errorFuncOutput = Some(ErrorFuncOutput(Some(efo))),
        waypoint = Some(Waypoint(nth = 100, ws => IO.File.write(ws, wps)))
      )
    )

    net.train(train.map(_._1), train.map(_._2))

    (train ++ test).foreach {
      case (x, y) =>
        val v = net(x)
        val c = v.data.indexOf(max(v))
        val t = y.data.indexOf(max(y))
        println(s"${classes(t)} classified as ${classes(c)}")
        println(net(x))
    }

    println(net)

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"Pos: $posWeights, Neg: $negWeights")

  }

}
