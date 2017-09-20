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
      |         Version 1.1.0
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
    val max = layers.map(_.neurons).max.toDouble
    val f = if (max > 10) 10.0 / max.toDouble else 1.0
    val center = math.ceil(((max * f) - 1.0) / 2.0)
    val cols = layers.map(l => (l.neurons - 1).toDouble * f).map { l =>
      val col = (0 until (max * f).toInt) map { _ => " " }
      col.zipWithIndex.map {
        case (c, i) if i <= center && i >= (center - math.ceil (l / 2.0))   => "O"
        case (c, i) if i >= center && i <= (center + math.floor(l / 2.0))   => "O"
        case (c, i)                                                         =>  c
      }
    }

    cols.reduce((l, r) => l.zip(r).map { case (a, b) => a + "         " + b }).foreach(l => println("         " + l))

    println()
    println()
    println()
  }

  def sayHi(): Unit = {
    println(welcome)
    if (settings.prettyPrint) prettyPrint()
  }

}
