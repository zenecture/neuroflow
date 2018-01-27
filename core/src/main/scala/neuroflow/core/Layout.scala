package neuroflow.core


import scala.annotation.{implicitNotFound, tailrec}

/**
  * @author bogdanski
  * @since 27.01.18
  */


sealed trait Layout extends Serializable {

  /** Prepends this layout with a new layer `head`. */
  def ::[H <: Layer](head: H): H :: this.type = neuroflow.core.::(head, tail = this)

}

case class ::[+H <: Layer, +T <: Layout](head: H, tail: T) extends Layout

case object Output extends Layout


trait ToList[L <: Layout, Target] {

  /** Converts this layout to a [[List]] casting to `Target`. */
  def apply(l: L): List[Target]

}

object ToList {

  implicit def toList[L <: Layout, T]: ToList[L, T] = new ToList[L, T] {

    def apply(l: L): List[T] = {
      val buffer = scala.collection.mutable.ListBuffer.empty[T]
      trav(l, head => buffer += head)
      buffer.toList
    }

    @tailrec private def trav(l: Layout, f: T => Unit): Unit = l match {
      case head :: tail =>
        f(head.asInstanceOf[T])
        trav(tail, f)
      case Output       =>
    }

  }

}


/**
  * Type-class witnessing that the first item within [[Layout]] `L` is `Predicate`.
  */
@implicitNotFound("Could not prove that this network starts with ${Predicate}. Check your architecture.")
trait StartsWith[L <: Layout, +Predicate]

object StartsWith {

  implicit def startsWith[H <: Layer, L <: Layout, H0]
    (implicit eq: H =:= H0): StartsWith[H :: L, H0] = new StartsWith[H :: L, H0] {}

}

/**
  * Type-class witnessing that the last item within [[Layout]] `L` is `Predicate`.
  */
@implicitNotFound("Could not prove that this network ends with ${Predicate}. Check your architecture.")
trait EndsWith[L <: Layout, +Predicate]

object EndsWith {

  implicit def hnil[P <: Layer]: (P :: Output.type) EndsWith P = new ((P :: Output.type) EndsWith P) {}

  implicit def hlist[H <: Layer, P, L <: Layout]
    (implicit e: L EndsWith P): (H :: L) EndsWith P = new ((H :: L) EndsWith P) {}

}
