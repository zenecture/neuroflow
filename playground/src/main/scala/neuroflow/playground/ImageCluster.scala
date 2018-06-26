package neuroflow.playground

import java.io.File

import breeze.linalg.DenseVector
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

//   val path = "/Users/felix/github/unversioned/objcat"
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

    val (inX, inY, inZ) = (200, 200, 3)
    val limit = 8192

    val objcat = new java.io.File(imagesS).list().map { d => imagesS + "/" + d }

    val train = objcat.take(limit).par.map { p =>
      val vec = loadVectorRGB(p).float
      vec -> vec
    }.seq

    val (f, g) = (ReLU, Linear)

    val in = Vector(inX * inY * inZ)
    val cl = Dense(inX * inY * inZ / 100, g)
    val id = Dense(inX * inY * inZ, f)

    val L = in :: cl :: id :: SquaredMeanError()

    val μ = 0.0

    implicit val breeder = neuroflow.core.WeightBreeder[Float].normal(μ, 0.001)
//    implicit val breeder = IO.File.weightBreeder[Float](wps + "-iter-400.nf")

    val net = Network(
      layout = L,
      Settings[Float](
        prettyPrint     = true,
        learningRate    = {
          case (i, α) if i < 150 => 1E-8
          case (i, α)            => 1E-9
        },
        updateRule      = Momentum(μ = 0.8f),
        iterations      = Int.MaxValue,
        precision       = 1E-4,
        batchSize       = Some(2048),
        gcThreshold     = Some(1024 * 1024 * 1024L /* 1G */),
        waypoint        = Some(Waypoint(nth = 50, (iter, ws) => IO.File.writeWeights(ws, wps + s"-iter-$iter.nf")))
      )
    )

    net.train(train.map(_._1), train.map(_._2))

    def write(source: Seq[(DenseVector[Float], DenseVector[Float])]) = {
      source.zipWithIndex.foreach { case (s, i) =>
        val vec = net(s._1).double
        val in = Image.imageFromVectorRGB(s._1.double, inX, inY)
        val out = Image.imageFromVectorRGB(vec, inX, inY)
        Image.writeImage(in, cluster + s"/$i-in.png", PNG)
        Image.writeImage(out, cluster + s"/$i-out.png", PNG)
      }
    }

    def find(source: Seq[(DenseVector[Float], DenseVector[Float])], n: Int, k: Int) = {
      val focused = net Ω cl
      val scores = source.map { s =>
        val vec = focused(s._1)
        s._1 -> vec
      }
      scores.take(n).zipWithIndex.foreach { case ((img, vec), i) =>
        val close = scores.map { iv =>
          (iv._1, iv._2, euclideanDistance(iv._2, vec))
        }.sortBy(_._3).drop(1).take(k)
        val target = Image.imageFromVectorRGB(img.double, inX, inY)
        Image.writeImage(target, cluster + s"/$i-target.png", PNG)
        close.foreach { ivs =>
          val m = Image.imageFromVectorRGB(ivs._1.double, inX, inY)
          Image.writeImage(m, cluster + s"/$i-match-${ivs._3}.png", PNG)
        }
      }
    }

//    write(train)
    find(Random.shuffle(train), 50, 5)

    val posWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ > 0.0).size)
    val negWeights = net.weights.foldLeft(0)((count, m) => count + m.findAll(_ < 0.0).size)

    println(s"|Weights| > 0 = $posWeights, |Weights| < 0 = $negWeights")

  }

}
