package neuroflow.common

/**
  * @author bogdanski
  * @since 02.08.17
  */
object Registry {

  private var _registry = 0

  def register(): String = _registry.synchronized {
    _registry += 1
    val s = "N" + _registry
    s
  }

}
