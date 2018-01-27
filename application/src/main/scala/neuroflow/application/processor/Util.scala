package neuroflow.application.processor

import scala.collection.immutable.Stream
import scala.util.Random

/**
  * @author bogdanski
  * @since 12.06.16
  */

object Util {

  /**
    * Strips given string `s` to only contain lower case letters.
    */
  def strip(s: String): String = s.replaceAll("[^a-zA-Z ]+", "").toLowerCase

  /**
    * Each element `x` of `xs` will be mapped to `xs` excluding this `x`.
    * Example:
    * (1,2,2,3) =>
    *   1 -> (2,2,3)
    *   2 -> (1,2,3)
    *   2 -> (1,2,3)
    *   3 -> (1,2,2)
    */
  def shuffle[T](xs: Seq[T]): Seq[(T, Seq[T])] =
    if(xs.size > 1) {
      val xz = xs.zipWithIndex
      xz.map { case (x, i) => x -> xz.filter(_._2 != i).map(_._1) }
    } else xs.map(x => x -> Seq(x))

  /**
    * Pretty prints given seq `v` with separator `sep` as a line record. (e.g. CSV)
    */
  def prettyPrint[A](v: Seq[A], sep: String = " "): String = v.foldLeft("")((l, r) => l + r + sep).dropRight(1)

  /**
    * Builds the cartesian of two traversables.
    */
  implicit class Crossable[X](xs: Traversable[X]) {
    def cross[Y](ys: Traversable[Y]): Traversable[(X, Y)] = for (x <- xs; y <- ys) yield (x, y)
  }

  /**
    * Builds a stream of random alpha chars.
    * (Variation from scala.util.Random)
    */
  def alpha(chars: String = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"): Stream[Char] = {
    def nextAlpha: Char = {
      chars.charAt(Random.nextInt(chars.length))
    }
    Stream.continually(nextAlpha)
  }

  /**
    * Builds a stream of random alpha chars, excluding those in `n`.
    */
  def alphaNot(n: String, chars: String = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"): Stream[Char] = {
    def nextAlpha: Char = {
      val cs = chars.filterNot(c => n.toUpperCase.contains(c))
      cs.charAt(Random.nextInt(cs.length))
    }
    Stream.continually(nextAlpha)
  }

}
