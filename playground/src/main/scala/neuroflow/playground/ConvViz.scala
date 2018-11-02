package neuroflow.playground

import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.Notation._
import neuroflow.application.processor.Image._
import neuroflow.core.Activators.Float._
import neuroflow.core._
import neuroflow.dsl.Convolution.autoTupler
import neuroflow.dsl.Implicits._
import neuroflow.dsl._
import neuroflow.nets.cpu.ConvNetwork._

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
      (s"glasses-$s", loadTensorRGB(path + "/glasses/" + s).float, ->(1.0f, 0.0f))
    }.seq

    val noglasses = new java.io.File(path + "/noglasses").list().map { s =>
      (s"noglasses-$s", loadTensorRGB(path + "/noglasses/" + s).float, ->(0.0f, 1.0f))
    }.seq

    val samples = Random.shuffle(glasses ++ noglasses)

    val f = ReLU

    val c0 = Convolution(dimIn = (400, 400, 3), padding = 1, field = 3, stride = 1, filters = 1, activator = f)
    val c1 = Convolution(dimIn = c0.dimOut,     padding = 1, field = 3, stride = 1, filters = 1, activator = f)
    val c2 = Convolution(dimIn = c1.dimOut,     padding = 1, field = 4, stride = 2, filters = 1, activator = f)
    val c3 = Convolution(dimIn = c2.dimOut,     padding = 1, field = 3, stride = 1, filters = 1, activator = f)

    val L = c0 :: c1 :: c2 :: c3 :: Dense(2, f) :: SoftmaxLogEntropy()

    val μ = 0

    implicit val weights = WeightBreeder[Float].normal(Map(
      0 -> (μ, 0.1),  1 -> (μ, 1), 2 -> (μ, 0.1), 3 -> (μ, 1), 4 -> (1E-4, 1E-4)
    ))

    val net = Network(
      layout = L,
      Settings[Float](
        prettyPrint     =  true,
        learningRate    =  { case (i, α) => 1E-3f },
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
      val cs = L.map { case c: Convolution[Float] => Some(c) case _ => None }.flatten.zipWithIndex
      samples.foreach {
        case (id, xs, ys) =>
          cs.foreach {
            case (c: Convolution[Float], ci: Int) =>
              val t = (net focus c).apply(xs)
              val is = imagesFromTensor3D(t.double, boost = 1.3)
              is.zipWithIndex.foreach { case (img, idx) => writeImage(img, path + s"/$stage" + s"/$ci-$idx-$id", PNG) }
          }
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