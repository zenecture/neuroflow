package neuroflow.nets

import neuroflow.core.Network

/**
  *
  * A shared network will constrain certain weights to have the same value.
  * This will lead to better performance for convoluted net architectures,
  * as well as more stable network with respect to translations and distortions
  * of an (still invariant) input.
  *
  *
  * @author bogdanski
  * @since 15.01.16
  */
trait SharedNetwork extends Network {

  /**
    *
    *
    * This is TODO.
    *
    */

}
