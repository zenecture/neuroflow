package neuroflow.playground

import neuroflow.application.preprocessor.Util._
import neuroflow.application.plugin.Style._
import neuroflow.core.Activator.Tanh
import neuroflow.core._
import neuroflow.nets.LBFGSNetwork._
import shapeless._

/**
  * @author bogdanski
  * @since 12.06.16
  */

object Hello {

  /*

        First audio recogniton with LBFGS network.

        Hello = (1.0, -1.0)
        Good-Bye = (-1.0, 1.0)

   */

  def prepare(s: String) = {
    val wav = getBytes(getFile(s)).map(_.toDouble).drop(44) // Drop WAV head data
    wav.map(_ / wav.max).grouped(5875).toList // 2*5*5*5*47 = 11750
  }

  def apply = {
    val fn = Tanh.apply
    val sets = Settings(maxIterations = 20, precision = 1E-4)
    val (a, b, c) = (prepare("audio/hello.wav"), prepare("audio/goodbye.wav"), prepare("audio/hello-alt.wav"))

    val nets = ((a zip b) zip c).par.map {
      case ((x, y), z) =>
        val net = Network(Input(x.size) :: Hidden(20, fn) :: Output(2, fn) :: HNil, sets)
        net.train(-->(x, y, z), -->(->(1.0, -1.0), ->(-1.0, 1.0), ->(1.0, -1.0)))
        (net, x, y, z)
    }

    val (r1, r2, r3) = nets map { case (n, x, y, z) => (n.evaluate(x), n.evaluate(y), n.evaluate(z)) } reduce { (l, r) =>
      val (a, b, c) = l
      val (u, v, w) = r
      val (x, y, z) = (a zip u, b zip v, c zip w)
      (x.map(l => l._1 + l._2), y.map(l => l._1 + l._2), z.map(l => l._1 + l._2))
    }

    val (hello, goodbye, helloalt) = (r1.map(_ / nets.size), r2.map(_ / nets.size), r3.map(_ / nets.size))

    println("hello.wav: " + hello)
    println("goodbye.wav: " + goodbye)
    println("hello-alt.wav: " + helloalt)
  }

}
