package neuroflow.playground

import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image
import neuroflow.application.processor.Image._
import neuroflow.core.Activators.Float._
import neuroflow.dsl.Convolution.IntTupler
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.dsl.Implicits._
import neuroflow.nets.gpu.ConvNetwork._

import scala.util.Random

/**
  * @author bogdanski
  * @since 12.03.18
  */


object ConvViz {

  /*

      Here we visualize activations of the convolutional layers before and after training.
      Each filter of a layer generates a gray-scale PNG image, which gives an insight how
      the net learns to separate the classes.

   */

  val path = "/Users/felix/github/unversioned/convviz"

  def apply = {

    val glasses = new java.io.File(path + "/glasses").list().map { s =>
      (s"glasses-$s", loadRgbTensor(path + "/glasses/" + s).float, ->(1.0f, 0.0f))
    }.seq

    val noglasses = new java.io.File(path + "/noglasses").list().map { s =>
      (s"noglasses-$s", loadRgbTensor(path + "/noglasses/" + s).float, ->(0.0f, 1.0f))
    }.seq

    val samples = Random.shuffle(glasses ++ noglasses)

    val f = ReLU

    val c0 = Convolution(dimIn = (400, 400, 3), padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 1, activator = f)
    val c1 = Convolution(dimIn = c0.dimOut,     padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 1, activator = f)
    val c2 = Convolution(dimIn = c1.dimOut,     padding = 1`²`, field = 4`²`, stride = 2`²`, filters = 1, activator = f)
    val c3 = Convolution(dimIn = c2.dimOut,     padding = 1`²`, field = 3`²`, stride = 1`²`, filters = 1, activator = f)

    val L = c0 :: c1 :: c2 :: c3 :: Dense(2, f) :: Softmax()

    val μ = 0

    implicit val breeder = neuroflow.core.WeightBreeder[Float].normal(Map(
      0 -> (μ, 0.1),  1 -> (μ, 1), 2 -> (μ, 0.1), 3 -> (μ, 1), 4 -> (1E-4, 1E-4)
    ))

    val net = Network(
      layout = L,
      Settings[Float](
        prettyPrint     =  true,
        learningRate    =  { case (i, α) => 1E-3 },
        updateRule      =  Momentum(μ = 0.8f),
        gcThreshold     =  Some(100 * 1024 * 1024L),
        batchSize       =  Some(20),
        iterations      =  250
      )
    )

    writeLayers(stage = "before")

    net.train(samples.map(_._2), samples.map(_._3))

    writeLayers(stage = "after")

    def writeLayers(stage: String): Unit = {
      samples.foreach {
        case (id, xs, ys) =>
          val f0 = (net Ω c0).apply(xs)
          val f1 = (net Ω c1).apply(xs)
          val f2 = (net Ω c2).apply(xs)
          val f3 = (net Ω c3).apply(xs)
          val i0s = Image.imagesFromTensor3D(c0.dimOut._1, c0.dimOut._2, f0.double, boost = 1.3)
          val i1s = Image.imagesFromTensor3D(c1.dimOut._1, c1.dimOut._2, f1.double, boost = 1.3)
          val i2s = Image.imagesFromTensor3D(c2.dimOut._1, c2.dimOut._2, f2.double, boost = 1.3)
          val i3s = Image.imagesFromTensor3D(c3.dimOut._1, c3.dimOut._2, f3.double, boost = 1.3)
          i0s.zipWithIndex.foreach { case (img, idx) => Image.writeImage(img, path + s"/$stage" + s"/c0-$idx-$id", PNG) }
          i1s.zipWithIndex.foreach { case (img, idx) => Image.writeImage(img, path + s"/$stage" + s"/c1-$idx-$id", PNG) }
          i2s.zipWithIndex.foreach { case (img, idx) => Image.writeImage(img, path + s"/$stage" + s"/c2-$idx-$id", PNG) }
          i3s.zipWithIndex.foreach { case (img, idx) => Image.writeImage(img, path + s"/$stage" + s"/c3-$idx-$id", PNG) }
      }
    }


  }

}

/*



             _   __                      ________
            / | / /__  __  ___________  / ____/ /___ _      __
           /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
          / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
         /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/
                                                            1.6.2


            Network : neuroflow.nets.gpu.ConvNetwork

            Weights : 80.061 (≈ 0,305408 MB)
          Precision : Single

               Loss : neuroflow.core.Softmax
             Update : neuroflow.core.Momentum

             Layout : 402*402*3 ~> [3*3 : 1*1] ~> 400*400*1 (ReLU)
                      402*402*1 ~> [3*3 : 1*1] ~> 400*400*1 (ReLU)
                      402*402*1 ~> [4*4 : 2*2] ~> 200*200*1 (ReLU)
                      202*202*1 ~> [3*3 : 1*1] ~> 200*200*1 (ReLU)
                      2 Dense (ReLU)






         O O O O O O O O O O      O O O O O O O O O O
         O O O O O O O O O O      O O O O O O O O O O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O          O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O          O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O
         O O O O O O O O O O      O O O O O O O O O O      O O O O O      O O O O O
         O O O O O O O O O O      O O O O O O O O O O
         O O O O O O O O O O      O O O O O O O O O O



 */