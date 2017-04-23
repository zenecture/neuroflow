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
      |         Version 0.602
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
      case h: HasActivator[_] => s"${h.neurons} (${h.activator.symbol})"
      case _ => l.neurons.toString
    }

  private def prettyPrint(): Unit = {
    val max = layers.map(_.neurons).max
    val center = (max - 1) / 2
    val cols = layers.map(l => l.neurons - 1).map { l =>
      val col = (0 until max) map { _ => " " }
      col.zipWithIndex.map {
        case (_, i) if i <= center && i >= center - (l / 2) => "O"
        case (_, i) if i >= center && i <= center + (l / 2) => "O"
        case (c, _) => c
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
