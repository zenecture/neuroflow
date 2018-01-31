package neuroflow.dsl

import neuroflow.core._
import neuroflow.dsl

import scala.annotation.{implicitNotFound, tailrec}

/**
  * @author bogdanski
  * @since 27.01.18
  */


/**
  * The layout describes the neural net as a linear graph
  * structure. It is implemented as a heterogenous list,
  * allowing compile-time checks for valid compositions.
  */
trait Layout extends Serializable {

  /** Prepends this layout with a new layer `head`. */
  def ::[H <: Layer](head: H): H :: this.type = dsl.::(head, tail = this)

}


case class ::[+H <: Layer, +T <: Layout](head: H, tail: T) extends Layout


trait Extractor[L <: Layout, Target, V] {

  /**
    * Extracts from [[Layout]] `l` a list
    * of type `Target` and the loss function.
    */
  def apply(l: L): (List[Target], LossFunction[V])

}

object Extractor {

  implicit def extractor[L <: Layout, T, V]: Extractor[L, T, V] = new Extractor[L, T, V] {

    def apply(l: L): (List[T], LossFunction[V]) = {
      val buffer = scala.collection.mutable.ListBuffer.empty[T]
      var loss: LossFunction[V] = null
      trav(l, head => buffer += head, L => loss = L)
      (buffer.toList, loss)
    }

    @tailrec private def trav(l: Layout, f: T => Unit, L: LossFunction[V] => Unit): Unit = l match {
      case head :: tail =>
        f(head.asInstanceOf[T])
        trav(tail, f, L)
      case l: LossFunction[V] => L(l)
    }

  }

}
