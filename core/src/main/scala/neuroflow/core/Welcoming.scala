package neuroflow.core

/**
  * @author bogdanski
  * @since 09.07.16
  */

trait Welcoming { self: Network =>

  private val welcome =
    s"""
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
      |         Version 0.701
      |
      |         Identifier: $identifier
      |         Network: ${this.getClass.getCanonicalName}
      |         Layout: ${layers.foldLeft("[")((s, l) => s + buildString(l) + ", ").dropRight(2) + "]"}
      |         Number of Weights: ${weights.map(_.size).sum}
      |
      |
      |
    """.stripMargin

  private def buildString(l: Layer): String =
    l match {
      case c: Convolutable    => c.reshape match {
        case Some(_)          => s"${c.filters} * ${c.fieldSize} ~> ${c.neurons} (${c.activator.symbol})"
        case None             => s"${c.filters} * ${c.fieldSize} (${c.activator.symbol})"
      }
      case h: HasActivator[_] => s"${h.neurons} ${l.symbol} (${h.activator.symbol})"
      case _                  => s"${l.neurons} ${l.symbol}"
    }

  private def prettyPrint(): Unit = {
    val max = layers.map(_.neurons).max
    val center = math.ceil((max.toDouble - 1.0) / 2.0)
    val cols = layers.map(l => (l.neurons - 1).toDouble).map { l =>
      val col = (0 until max) map { _ => " " }
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
