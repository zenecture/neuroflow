package neuroflow.core

import java.text.NumberFormat.{ getIntegerInstance => formatter }

/**
  * @author bogdanski
  * @since 09.07.16
  */

trait Welcoming { self: Network[_, _] =>

  private val welcome =
    f"""
      |
      |
      |
      |             _   __                      ________
      |            / | / /__  __  ___________  / ____/ /___ _      __
      |           /  |/ / _ \\/ / / / ___/ __ \\/ /_  / / __ \\ | /| / /
      |          / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
      |         /_/ |_/\\___/\\__,_/_/   \\____/_/   /_/\\____/|__/|__/
      |
      |
      |         Version 1.1.1
      |
      |         Identifier: $identifier
      |         Network: ${this.getClass.getCanonicalName}
      |         Layout: ${layers.foldLeft("[")((s, l) => s + buildString(l) + ", ").dropRight(2) + "]"}
      |         Number of Weights: ${ formatter.format(weights.map(_.size).sum) } (≈ ${ weights.map(_.size).sum.toDouble * 8.0 / 1024.0 / 1024.0 }%.6g MB)
      |
      |
      |
    """.stripMargin

  private def buildString(l: Layer): String =
    l match {
      case c: Convolution     => s"${c.dimIn._1}*${c.dimIn._2}*${c.dimIn._3} ~> ${c.dimOut._1}*${c.dimOut._2}*${c.dimOut._3} (${c.activator.symbol})"
      case h: HasActivator[_] => s"${h.neurons} ${l.symbol} (${h.activator.symbol})"
      case _                  => s"${l.neurons} ${l.symbol}"
    }

  private def prettyPrint(): Unit = {
    val max = layers.map {
      case Convolution(dimIn, _, _, _, _) => math.max(dimIn._1, dimIn._2)
      case l: Layer                       => l.neurons
    }.max
    val f = if (max > 10) 10.0 / max.toDouble else 1.0
    val potency = layers.flatMap {
      case Convolution(dimIn, _, _, _, _) =>
        val m = (1 to (dimIn._1.toDouble * f).toInt).map { _ => (dimIn._2, true, true) }
        val s = m.dropRight(1) :+ (m.last._1, m.last._2, false)
        s
      case l: Layer                       => Seq((l.neurons, false, false))
    }
    val center = math.ceil(((max * f) - 1.0) / 2.0)
    val cols = potency.map(p => ((p._1 - 1).toDouble * f, p._2, p._3)).map { l =>
      val col = (0 until (max * f).toInt) map { _ => " " }
      col.zipWithIndex.map {
        case (c, i) if i <= center && i >= (center - math.ceil (l._1 / 2.0))   => ("O", l._2, l._3)
        case (c, i) if i >= center && i <= (center + math.floor(l._1 / 2.0))   => ("O", l._2, l._3)
        case (c, i)                                                            => ( c , l._2, l._3)
      }
    }

    cols.reduce((l, r) => l.zip(r).map {
      case ((a, true, true), (b, true, true))   => (a + " " + b,           true,  true)
      case ((a, true, true), (b, true, false))  => (a + " " + b + "     ", true,  true)
      case ((a, _, _), (b, _, _))               => (a + "     " + b,       false, false)
    }).foreach(l => println("         " + l._1))

    println()
    println()
    println()
  }

  def sayHi(): Unit = {
    println(welcome)
    if (settings.prettyPrint) prettyPrint()
  }

}
