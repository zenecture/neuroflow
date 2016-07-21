package neuroflow.core

/**
  * @author bogdanski
  * @since 09.07.16
  */

trait Welcoming { self: Network =>

  val welcome =
    s"""
      |
      |
      |
      |       _   __                      ________
      |      / | / /__  __  ___________  / ____/ /___ _      __
      |     /  |/ / _ \\/ / / / ___/ __ \\/ /_  / / __ \\ | /| / /
      |    / /|  /  __/ /_/ / /  / /_/ / __/ / / /_/ / |/ |/ /
      |   /_/ |_/\\___/\\__,_/_/   \\____/_/   /_/\\____/|__/|__/
      |
      |
      |         Version 0.3
      |
      |         Network: ${this.getClass.getCanonicalName}
      |         Layout: ${layers.foldLeft("[")((s, l) => s + buildString(l) + ", ").dropRight(2) + "]"}
      |         Number of Weights: ${weights.map(_.size).sum}
      |
      |
      |
    """.stripMargin

  private def buildString(l: Layer) =
    l match {
      case h: HasActivator[_] => s"${h.neurons} (${h.activator.name})"
      case i => l.neurons.toString
    }

  print(welcome)

}
