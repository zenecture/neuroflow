package neuroflow.playground

import neuroflow.application.plugin.Extensions._
import neuroflow.application.plugin.IO.Jvm._
import neuroflow.application.plugin.Notation._
import neuroflow.core.Activator.Tanh
import neuroflow.core._
import neuroflow.dsl._
import neuroflow.nets.cpu.DenseNetwork._

/**
  * @author bogdanski
  * @since 12.06.16
  */

object AudioRecognition {

  /*

        Audio wave classification

        Hello = (1.0, -1.0)
        Good-Bye = (-1.0, 1.0)

   */

  def prepare(s: String) = {
    val wav = getBytes(getResourceFile(s)).map(_.toDouble).toVector.drop(44) // Drop WAV head data
    wav.map(_ / wav.max).grouped(2*5*5*5).toVector // 2*5*5*5*47 = 11750
  }

  def apply = {

    implicit val wp = neuroflow.core.WeightProvider.FFN[Double].random(-1, 1)

    val fn = Tanh
    val sets = Settings[Double](iterations = 2000, precision = 1E-4)
    val (a, b, c) = (prepare("audio/hello.wav"), prepare("audio/goodbye.wav"), prepare("audio/hello-alt.wav"))

    val nets = ((a zip b) zip c).par.map {
      case ((x, y), z) =>
        val net = Network(Vector(x.size) :: Dense(20, fn) :: Dense(2, fn) :: SquaredMeanError(), sets)
        net.train(Array(x.denseVec, y.denseVec, z.denseVec), Array(->(1.0, -1.0), ->(-1.0, 1.0), ->(1.0, -1.0)))
        (net, x, y, z)
    }

    val (r1, r2, r3) = nets map { case (n, x, y, z) => (n.evaluate(x.denseVec), n.evaluate(y.denseVec), n.evaluate(z.denseVec)) } reduce { (l, r) =>
      val (a, b, c) = l
      val (u, v, w) = r
      (a + u, b + v, c + w)
    }

    val (hello, goodbye, helloalt) = (r1.map(_ / nets.size), r2.map(_ / nets.size), r3.map(_ / nets.size))

    println("hello.wav: " + hello)
    println("goodbye.wav: " + goodbye)
    println("hello-alt.wav: " + helloalt)
  }

}


/*

      Layout:
        Hello ->(1.0, -1.0)
        Goodbye ->(-1.0, 1.0)

      [scala-execution-context-global-84] INFO neuroflow.nets.DefaultNetwork - [31.07.2017 21:41:07:016] Took 2000 iterations of 2000 with Mean Error = 0,000153440
      hello.wav: Vector(0.9960103806540942, -0.9965630625015153)
      goodbye.wav: Vector(-0.9871920187975733, 0.9034866666916598)
      hello-alt.wav: Vector(0.987595444420504, -0.9462879943364947)
      [success] Total time: 36 s, completed 31.07.2017 21:41:07



 */
