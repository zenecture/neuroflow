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
      |         Version 0.301-SNAPSHOT
      |
      |         Layout: ${layers.foldLeft("[")((s, l) => s + l.neurons + ", ").dropRight(2) + "]"}
      |         Network: ${this.getClass.getCanonicalName}
      |         Number of Weights: ${weights.map(_.size).sum}
      |
      |
      |
    """.stripMargin

  print(welcome)

}
