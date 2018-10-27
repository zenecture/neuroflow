package neuroflow.playground

import java.io.File

import breeze.linalg.functions.euclideanDistance
import javax.imageio.ImageIO
import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO
import neuroflow.application.plugin.IO._
import neuroflow.application.processor.Image
import neuroflow.application.processor.Image._
import neuroflow.core.Activators.Float._
import neuroflow.core._
import neuroflow.dsl.Implicits._
import neuroflow.dsl._
import neuroflow.nets.gpu.DenseNetwork._

import scala.util.Random

/**
  * @author bogdanski
  * @since 24.06.18
  */
object ImageCluster {

//  val path = "/Users/felix/github/unversioned/objcat"
  val path = "/home/felix"
  val images = path + "/101_ObjectCategories"
  val imagesS = path + "/scaled"
  val cluster = path + "/cluster"
  val wps  = path + "/waypoint"
  val lfo  = path + "/lfo.txt"

  def scale = {

    println("Scaling images ...")

    val objcat = new java.io.File(images).list().filterNot(_.startsWith(".")).flatMap { d =>
      val p = images + "/" + d
      new java.io.File(p).list().filterNot(_.startsWith(".")).map(d + "/" + _)
    }

    objcat.foreach { p =>
      val img = ImageIO.read(new File(images + "/" + p))
      val scaled = Image.scale(img, 200, 200)
      Image.writeImage(scaled, imagesS + "/" + p.replace("/", "_"), JPG)
    }

  }

  def apply = {

    println("Loading data ...")

    val classes = Seq("airplanes", "ewer", "dragonfly")
    val (inX, inY, inZ) = (200, 200, 3)
    val (limit, samples, topK) = (8196, 10, 5)

    val objcat = new java.io.File(imagesS).list().map { d => imagesS + "/" + d }

    val train = objcat.take(limit).filter(p => classes.exists(p.contains)).par.map { p =>
      val vec = loadVectorRGB(p).float
      vec -> classes.find(p.contains).get
    }.seq

    val (f, g) = (ReLU, Linear)

    val in = Vector(inX * inY * inZ)
    val cl = Dense(inX * inY * inZ / 100, g)
    val id = Dense(inX * inY * inZ, f)

    val L = in :: cl :: id :: SquaredError()

    val μ = 0.0

    implicit val weights = WeightBreeder[Float].normal(μ, 0.001)
//    implicit val weights = IO.File.weightBreeder[Float](wps + "-iter-6400.nf")

    val net = Network(
      layout = L,
      Settings[Float](
        prettyPrint     = true,
        learningRate    = {
          case (i, α) if i < 150 => 1E-8f
          case (i, α)            => 1E-9f
        },
        updateRule      = Momentum(μ = 0.8f),
        iterations      = Int.MaxValue,
        precision       = 1E-4,
        batchSize       = Some(train.size),
        gcThreshold     = Some(1024 * 1024 * 1024L /* 1G */),
        waypoint        = Some(Waypoint(nth = 100, (iter, ws) => IO.File.writeWeights(ws, wps + s"-iter-$iter.nf")))
      )
    )

    net.train(train.map(_._1), train.map(_._1))

    println("Generating matches ...")

    val focused = net focus cl
    val targets = Random.shuffle(train).take(samples)

    targets.zipWithIndex.par.foreach { case ((t, c), idx) =>
      val score = focused(t)
      val ed = train.filter(_._2 == c).map(tc => tc._1 -> euclideanDistance(tc._1, t)).sortBy(_._2).take(topK)
      val nn = train.filter(_._2 == c).map(tc => tc._1 -> euclideanDistance(focused(tc._1), score)).sortBy(_._2).take(topK)
      ed.foreach { i => Image.writeImage(Image.imageFromVectorRGB(i._1.double, inX, inY), cluster + s"/ed/$idx-${i._2}.png", PNG) }
      nn.foreach { i => Image.writeImage(Image.imageFromVectorRGB(i._1.double, inX, inY), cluster + s"/nn/$idx-${i._2}.png", PNG) }
    }

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"|Weights| > 0 = $posWeights, |Weights| < 0 = $negWeights")

  }

}

/*

Run example (1-20):
Loading data ...




             _   __                      ________
            / | / /__  __  ___________  / ____/ /___ _      __
           /  |/ / _ \/ / / / ___/ __ \/ /_  / / __ \ | /| / /
          / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
         /_/ |_/\___/\__,_/_/   \____/_/   /_/\____/|__/|__/
                                                            1.6.6


            Network : neuroflow.nets.gpu.DenseNetwork

            Weights : 288,000,000 (≈ 1098.63 MB)
          Precision : Single

               Loss : neuroflow.core.SquaredMeanError
             Update : neuroflow.core.Momentum

             Layout : 120000 Vector
                      1200 Dense (x)
                      120000 Dense (ReLU)






         O           O
         O           O
         O           O
         O           O
         O     O     O
         O     O     O
         O           O
         O           O
         O           O
         O           O



 */