package neuroflow.core

import neuroflow.core.Network.LearningRate

import scala.collection.{Map, Set}

/**
  * @author bogdanski
  * @since 03.01.16
  */


/**
  *
  * Settings of a neural network, where:
  *
  *   `verbose`             Indicates logging behavior on console.
  *   `learningRate`        A function from iteration `i` and learning rate `α`, producing a new learning rate.
  *   `updateRule`          Defines the relationship between gradients, weights and learning rate during training.
  *   `precision`           Training stops if loss is small enough.
  *   `iterations`          Training stops after `iterations`.
  *   `prettyPrint`         If true, the layout is rendered graphically on console.
  *   `batchSize`           Number of samples presented per weight update. (1=on-line, ..., n=full-batch)
  *   `gcThreshold`         Garbage Collection threshold for GPU, set in bytes.
  *   `lossFuncOutput`      Prints the loss to the specified file/closure.
  *   `waypoint`            Waypoint function, e. g. to periodically save weights.
  *   `approximation`       If set, gradients are approximated numerically.
  *   `regularization`      The respective regulator tries to avoid over-fitting.
  *   `partitions`          A sequential training sequence can be partitioned for RNNs. (0 index-based)
  *   `specifics`           Some nets use specific parameters set in the `specifics` map.
  *
  */
case class Settings[V]
                    (verbose           :  Boolean                      =  true,
                     learningRate      :  LearningRate[V]              =  { case (i, α) => α }: LearningRate[V],
                     updateRule        :  Update[V]                    =  Vanilla[V](),
                     precision         :  Double                       =  1E-3,
                     iterations        :  Int                          =  Int.MaxValue,
                     prettyPrint       :  Boolean                      =  false,
                     batchSize         :  Option[Int]                  =  None,
                     gcThreshold       :  Option[Long]                 =  None,
                     lossFuncOutput    :  Option[LossFuncOutput]       =  None,
                     waypoint          :  Option[Waypoint[V]]          =  None,
                     approximation     :  Option[Approximation[V]]     =  None,
                     regularization    :  Option[Regularization]       =  None,
                     partitions        :  Option[Set[Int]]             =  None,
                     specifics         :  Option[Map[String, Double]]  =  None) extends Serializable

