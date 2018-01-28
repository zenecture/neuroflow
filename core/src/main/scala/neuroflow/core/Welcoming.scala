package neuroflow.core

import java.text.NumberFormat.{ getIntegerInstance => formatter }

/**
  * @author bogdanski
  * @since 09.07.16
  */

trait Welcoming { self: Network[_, _, _] =>

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
      |            Version : 1.4.6
      |
      |            Network : $identifier
      |               Loss : ${self.lossFunction.getClass.getCanonicalName}
      |             Update : ${self.settings.updateRule.getClass.getCanonicalName}
      |
      |             Layout : ${layers.foldLeft("")((s, l) => s + buildString(l) + "\n                      ").dropRight(2)}
      |            Weights : ${ formatter.format(weights.map(_.size).sum) } (≈ ${ weights.map(_.size).sum.toDouble * sizeOf(numericPrecision) / 1024.0 / 1024.0 }%.6g MB)
      |          Precision : $numericPrecision
      |
      |
      |
    """.stripMargin

  private def buildString(l: Layer): String =
    l match {
      case c:  Convolution[_] => s"${c.dimInPadded._1}*${c.dimInPadded._2}*${c.dimInPadded._3} ~> ${c.dimOut._1}*${c.dimOut._2}*${c.dimOut._3} (${c.activator.symbol})"
      case h: HasActivator[_] => s"${h.neurons} ${l.symbol} (${h.activator.symbol})"
      case _                  => s"${l.neurons} ${l.symbol}"
    }

  private def sizeOf(p: String): Double = p match {
    case "Double"           => TypeSize.TypeSizeDouble()
    case "Float" | "Single" => TypeSize.TypeSizeFloat()
    case "Integer"          => TypeSize.TypeSizeInt()
  }

  private def prettyPrint(): Unit = {

    val max = layers.map {
      case c: Convolution[_] => math.max(math.max(c.dimInPadded._1, c.dimInPadded._2), math.max(c.dimOut._1, c.dimOut._2))
      case l: Layer          => l.neurons
    }.max

    val f = if (max > 10) 10.0 / max.toDouble else 1.0

    val potency = layers.zipWithIndex.flatMap {
      case (c: Convolution[_], _)            => Seq( /* field, stride: prevent sanity checks */
//                                                  c.copy(dimIn = c.dimInPadded, field = (1, 1), stride = (1, 1)),
                                                  c.copy(dimIn = c.dimOut, field = (1, 1), stride = (1, 1))
                                                )
      case (l: Layer, _)                     => Seq(l)
    }.flatMap {
      case Convolution(dimIn, _, _, _, _, _) =>
        val m = (1 to math.ceil(dimIn._1.toDouble * f).toInt).map { _ => (dimIn._2, true, true) }
        val s = m.dropRight(1) :+ (m.last._1, m.last._2, false)
        s
      case l: Layer => Seq((l.neurons, false, false))
    }

    val center = math.ceil(((max * f) - 1.0) / 2.0)

    val  cols  = potency.map(p => ((p._1 - 1).toDouble * f, p._2, p._3)).map { l =>
      val  col = (0 until (max * f).toInt) map { _ => " " }
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
