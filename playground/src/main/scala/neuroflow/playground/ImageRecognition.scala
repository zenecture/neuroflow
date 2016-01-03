package neuroflow.playground

import java.io.File

import neuroflow.core.Activator.Sigmoid
import neuroflow.core.{Network, Input, Hidden, Output}
import neuroflow.application.classification.Image._

/**
  * @author bogdanski
  * @since 03.01.16
  */
object ImageRecognition {

  def getImg(path: String): File = new File(getClass.getClassLoader.getResource(path).toURI)

  def apply = {
    // Training
    val plus = extractRgb(getImg("img/plus.png")).grouped(30).toList
    val heart = extractRgb(getImg("img/heart.png")).grouped(30).toList

    // Testing
    val heartDistorted = extractRgb(getImg("img/heart_distorted.png")).grouped(30).toList
    val heartRotated = extractRgb(getImg("img/heart_rotated.png")).grouped(30).toList
    val plusRotated = extractRgb(getImg("img/plus_rotated.png")).grouped(30).toList
    val random = extractRgb(getImg("img/random.png")).grouped(30).toList

    println(s"Training ${plus.size + heart.size} samples...")

    val training = plus.zip(heart).zip(random)
    val nets = training.map { sample => // One could parallelize this to gain significant performance boosts
      val (ph, r) = sample
      val (p, h) = ph
      val net = Network(Input(p.size) :: Hidden(20, Sigmoid.apply) :: Output(3, Sigmoid.apply) :: Nil)
      net.train(Seq(p, h, r), Seq(Seq(1.0, 0.0, 0.0), Seq(0.0, 1.0, 0.0), Seq(0.0, 0.0, 1.0)), 20.0, 0.001, 1000)
      net
    }

    def eval(image: Seq[Seq[Double]]) = {
      image.zip(nets).map { s =>
        val (xs, net) = s
        net.evaluate(xs)
      } reduce { (a, b) =>
        a.zip(b).map(l => l._1 + l._2)
      } map { end => end / nets.size }
    }

    val plusResult = eval(plus)
    val plusRotatedResult = eval(plusRotated)
    val heartResult = eval(heart)
    val heartDistortedResult = eval(heartDistorted)
    val heartRotatedResult = eval(heartRotated)
    val randomResult = eval(random)

    println(s"Plus classified: $plusResult")
    println(s"Plus Rotated classified: $plusRotatedResult")
    println(s"Heart classified: $heartResult")
    println(s"Heart distorted classified: $heartDistortedResult")
    println(s"Heart rotated classified: $heartRotatedResult")
    println(s"Random classified: $randomResult")
  }

}
