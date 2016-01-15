package neuroflow.common

import java.io._

/**
  * @author bogdanski
  * @since 09.01.16
  */

object ByteLevel {

  /**
    * Serializes `anyRef` to byte array.
    */
  def serialize(anyRef: AnyRef): Array[Byte] = synchronized { ~> (new ByteArrayOutputStream) map
    (stream => (stream, new ObjectOutputStream(stream))) io (_._2.writeObject(anyRef)) io (_._1.close) io (_._2.close) map (_._1.toByteArray)
  }

  /**
    * Deserializes `bytes` to `T`.
    */
  def deserialize[T](bytes: Array[Byte]): T = synchronized { ~> (new ByteArrayInputStream(bytes)) map
    (new CustomObjectReader(Thread.currentThread.getContextClassLoader, _)) map (r => (r.readObject.asInstanceOf[T], r)) io (_._2.close) map (_._1)
  }

}

class CustomObjectReader(cl: ClassLoader, bai: InputStream) extends ObjectInputStream(bai) {
  override def resolveClass(desc: ObjectStreamClass): Class[_] = cl.loadClass(desc.getName)
}