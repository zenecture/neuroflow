package neuroflow.cuda

import jcuda.driver.{JCudaDriver, CUdevice}

/**
  * @author dlwh
  **/
class CuDevice(val device: CUdevice) {
  override def toString = device.toString
}

object CuDevice {

  implicit lazy val defaultDevice = apply(0)

  def apply(deviceNumber: Int): CuDevice = {
    JCudaDriver.cuInit(0)
    JCudaDriver.setExceptionsEnabled(true)
    val device = new CUdevice
    JCudaDriver.cuDeviceGet(device, deviceNumber)
    new CuDevice(device)
  }

}
