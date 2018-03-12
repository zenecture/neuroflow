package neuroflow.playground

import breeze.linalg.{DenseVector, max}
import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.common.~>
import neuroflow.core.Activators.Float._
import neuroflow.dsl.Convolution.IntTupler
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.nets.gpu.ConvNetwork._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def apply = {

//    val path = "/Users/felix/github/unversioned/cifar"
    val path = "/home/felix"
    val wps  = path + "/waypoint"
    val lfo  = path + "/lfo.txt"

    val classes = Seq("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    val classVecs = classes.zipWithIndex.map { case (c, i) => c -> ~>(ζ[Float](classes.size)).io(_.update(i, 1.0f)).t }.toMap

    println("Loading data ...")

    val limits = (50000, 10000)

    val train = new java.io.File(path + "/train").list().take(limits._1).par.map { s =>
      val c = classes.find(z => s.contains(z)).get
      loadRgbTensor(path + "/train/" + s).float -> classVecs(c)
    }.seq

    val test = new java.io.File(path + "/test").list().take(limits._2).par.map { s =>
      val c = classes.find(z => s.contains(z)).get
      loadRgbTensor(path + "/test/" + s).float -> classVecs(c)
    }.seq

    classes.foreach { c =>
      println(s"|$c| = " + train.count(l => l._2 == classVecs(c)))
    }

    val f = ReLU

    val c0 = Convolution(dimIn = (32, 32, 3),  padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 128, activator = f)
    val c1 = Convolution(dimIn = c0.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 128, activator = f)
    val c2 = Convolution(dimIn = c1.dimOut,    padding = 1`²`, field = 4`²`, stride = 2`²`, filters = 128, activator = f)
    val c3 = Convolution(dimIn = c2.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 128, activator = f)
    val c4 = Convolution(dimIn = c3.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 128, activator = f)
    val c5 = Convolution(dimIn = c4.dimOut,    padding = 1`²`, field = 4`²`, stride = 2`²`, filters = 128, activator = f)
    val c6 = Convolution(dimIn = c5.dimOut,    padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 128, activator = f)
    val c7 = Convolution(dimIn = c6.dimOut,    padding = 0`²`, field = 1`²`, stride = 1`²`, filters = 128, activator = f)
    val c8 = Convolution(dimIn = c7.dimOut,    padding = 0`²`, field = 1`²`, stride = 1`²`, filters =  10, activator = f)

    val L = c0 :: c1 :: c2 :: c3 :: c4 :: c5 :: c6 :: c7 :: c8 :: Dense(10, f) :: Softmax()

    val μ = 0.0

    implicit val breeder = neuroflow.core.WeightBreeder[Float].normal(Map(
      0 -> (μ, 0.1),  1 -> (μ, 0.1), 2 -> (μ, 0.1),
      3 -> (μ, 0.01), 4 -> (μ, 0.1), 5 -> (μ, 0.01),
      6 -> (μ, 0.01), 7 -> (μ, 0.1), 8 -> (μ, 1.0),
      9 -> (0.01, 0.01)
    ))

//    implicit val breeder = neuroflow.application.plugin.IO.File.weightBreeder[Float](wps + "-iter-1000.nf")

    val net = Network(
      layout = L,
      Settings[Float](
        prettyPrint     = true,
        learningRate    = {
          case (i, α) if i < 4000 => 1E-5
          case (i, α)             => 1E-6
        },
        updateRule      = Momentum(μ = 0.8f),
        iterations      = Int.MaxValue,
        precision       = 1E-2,
        batchSize       = Some(250),
        gcThreshold     = Some(1024 * 1024 * 1024L /* 1G */),
        lossFuncOutput  = Some(LossFuncOutput(Some(lfo))),
        waypoint        = Some(Waypoint(nth = 1000, (iter, ws) => File.writeWeights(ws, wps + s"-iter-$iter.nf")))
      )
    )

    net.train(train.map(_._1), train.map(_._2))

    def eval(source: Seq[(Tensor3D[Float], DenseVector[Float])], s: String) = {
      var i = 1
      val rate = source.map {
        case (x, y) =>
          val v = net(x)
          val c = v.toArray.indexOf(max(v))
          val t = y.toArray.indexOf(max(y))
          println(s"Step $i.")
          i += 1
          if (c == t) 1.0 else 0.0
      }.sum / source.size.toDouble
      println(s"$s: Recognition rate = ${rate * 100.0} %, Error rate = ${(1.0 - rate) * 100.0} %!")
    }

    eval(train, "train")
    eval(test, "test")
    eval(train ++ test, "train ++ test")

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"|Weights| > 0 = $posWeights, |Weights| < 0 = $negWeights")

  }

}



/*



             _   __                      ________
            / | / /__  __  ___________  / ____/ /___ _      __
           /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
          / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
         /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/
                                                            1.5.8


            Network : neuroflow.nets.gpu.ConvNetwork

            Weights : 1,141,632 (≈ 4.35498 MB)
          Precision : Single

               Loss : neuroflow.core.Softmax
             Update : neuroflow.core.Momentum

             Layout : 34*34*3 ~> [3*3 : 1*1] ~> 32*32*128 (R)
                      34*34*128 ~> [3*3 : 1*1] ~> 32*32*128 (R)
                      34*34*128 ~> [4*4 : 2*2] ~> 16*16*128 (R)
                      18*18*128 ~> [3*3 : 1*1] ~> 16*16*128 (R)
                      18*18*128 ~> [3*3 : 1*1] ~> 16*16*128 (R)
                      18*18*128 ~> [4*4 : 2*2] ~> 8*8*128 (R)
                      10*10*128 ~> [3*3 : 1*1] ~> 8*8*128 (R)
                      8*8*128 ~> [1*1 : 1*1] ~> 8*8*128 (R)
                      8*8*128 ~> [1*1 : 1*1] ~> 8*8*10 (R)
                      10 Dense (R)






         O O O O O O O O O O      O O O O O O O O O O
         O O O O O O O O O O      O O O O O O O O O O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O      O O O O O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O      O O O O O      O O O      O O O      O O O      O O O          O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O      O O O O O      O O O      O O O      O O O      O O O          O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O      O O O O O      O O O      O O O      O O O      O O O          O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O      O O O O O      O O O      O O O      O O O      O O O          O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O      O O O O O
         O O O O O O O O O O      O O O O O O O O O O
         O O O O O O O O O O      O O O O O O O O O O




         The CIFAR set has 50000 (+10000 test) images á 32*32*3 pixels (rgb) from 10 classes.
         All image tensors are loaded from raw PNGs, no pre-processing (contrast enhancing et al) is applied.

         After 2-3h on a nVidia Tesla P100 GPU, the model gives these results:

           train: Recognition rate = 85.71 %, Error rate = 14.29 %!
           test: Recognition rate = 70.27 %, Error rate = 29.73 %!
           train ++ test: Recognition rate = 83.13 %, Error rate = 16.87 %!
           |Weights| > 0 = 572125, |Weights| < 0 = 569507

         Then, the net fits on the train set in the 94 % area, while
         the recognition rate plateaus around 70 % on the test set:

           train: Recognition rate = 94.26 %, Error rate = 5.74 %!
           test: Recognition rate = 69.04 %, Error rate = 30.96 %!
           train ++ test: Recognition rate = 90.05 %, Error rate = 9.95 %!
           |Weights| > 0 = 572845, |Weights| < 0 = 568787


 */



