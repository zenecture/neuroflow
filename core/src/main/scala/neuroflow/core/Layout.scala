package neuroflow.core


import scala.annotation.{implicitNotFound, tailrec}

/**
  * @author bogdanski
  * @since 27.01.18
  */


/**
  * The layout describes the neural flow as a linear graph
  * structure. It is implemented as a heterogenous list,
  * allowing compile-time checks for valid compositions.
  */
trait Layout extends Serializable {

  /** Prepends this layout with a new layer `head`. */
  def ::[H](head: H): H :: this.type = neuroflow.core.::(head, tail = this)

}


case class ::[+H, +T <: Layout](head: H, tail: T) extends Layout


trait Extractor[L <: Layout, Target, V] {

  /**
    * Extracts from [[Layout]] `l` a list
    * of type `Target` and the loss function.
    *
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


/**
  * Type-class witnessing that the first item within [[Layout]] `L` is `Predicate`.
  */
@implicitNotFound("Could not prove that this network starts with ${Predicate}. Check your layout: ${L}")
trait StartsWith[L <: Layout, +Predicate]

object StartsWith {

  implicit def startsWith[H <: Layer, L <: Layout, H0]
    (implicit eq: H =:= H0): StartsWith[H :: L, H0] = new StartsWith[H :: L, H0] {}

}

/**
  * Type-class witnessing that the last item within [[Layout]] `L` is `Predicate`.
  */
@implicitNotFound("Could not prove that this network ends with ${Predicate}. Check your layout: ${L}")
trait EndsWith[L <: Layout, +Predicate]

object EndsWith {

  implicit def hsme[P, V]: (P :: SquaredMeanError[V]) EndsWith P = new ((P :: SquaredMeanError[V]) EndsWith P) { }

  implicit def hsmx[P, V]: (P :: Softmax[V]) EndsWith P = new ((P :: Softmax[V]) EndsWith P) { }

  implicit def hlist[H <: Layer, P, L <: Layout]
    (implicit e: L EndsWith P): (H :: L) EndsWith P = new ((H :: L) EndsWith P) { }

}
