package neuroflow.nets

/**
  * @author bogdanski
  * @since 02.08.17
  */
private[nets] object Registry {

  private var _registry = 0

  def register(): String = synchronized {
    _registry += 1
    val s = "N" + _registry
    s
  }

}
