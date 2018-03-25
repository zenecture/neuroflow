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
  *   `updateRule`          Defines the relationship between gradient, weights and learning rate during training.
  *   `precision`           The training will stop if precision is high enough.
  *   `iterations`          The training will stop if maximum iterations is reached.
  *   `prettyPrint`         If true, the layout is rendered graphically on console.
  *   `coordinator`         The coordinator host address for distributed training.
  *   `transport`           Transport throughput specifics for distributed training.
  *   `parallelism`         Controls how many threads are used for distributed training.
  *   `batchSize`           Controls how many samples are presented per weight update. (1=on-line, ..., n=full-batch)
  *   `gcThreshold`         Fine tune GC for GPU, the threshold is set in bytes
  *   `lossFuncOutput`      Prints the loss to the specified file/closure.
  *   `waypoint`            Periodic actions can be executed, e.g. saving the weights every n steps.
  *   `approximation`       If set, the gradients are approximated numerically.
  *   `regularization`      The respective regulator tries to avoid over-fitting.
  *   `partitions`          A sequential training sequence can be partitioned for RNNs. (0 index-based)
  *   `specifics`           Some nets use specific parameters set in the `specifics` map.
  *
  */
case class Settings[V]
                    (verbose           :  Boolean                      =  true,
                     learningRate      :  LearningRate                 =  { case (i, α) => α },
                     updateRule        :  Update[V]                    =  Vanilla[V](),
                     precision         :  Double                       =  1E-5,
                     iterations        :  Int                          =  100,
                     prettyPrint       :  Boolean                      =  true,
                     coordinator       :  Node                         =  Node("localhost", 2552),
                     transport         :  Transport                    =  Transport(100000, "128 MiB"),
                     parallelism       :  Option[Int]                  =  Some(Runtime.getRuntime.availableProcessors),
                     batchSize         :  Option[Int]                  =  None,
                     gcThreshold       :  Option[Long]                 =  None,
                     lossFuncOutput    :  Option[LossFuncOutput]       =  None,
                     waypoint          :  Option[Waypoint[V]]          =  None,
                     approximation     :  Option[Approximation[V]]     =  None,
                     regularization    :  Option[Regularization]       =  None,
                     partitions        :  Option[Set[Int]]             =  None,
                     specifics         :  Option[Map[String, Double]]  =  None) extends Serializable

