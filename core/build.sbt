libraryDependencies ++= Seq(
  "org.specs2" %% "specs2-core" % "3.6.4" % "test",
  "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test"
)

scalacOptions in ThisBuild ++= Seq("-feature", "-deprecation", "-language:postfixOps", "-language:higherKinds", "-language:implicitConversions")

scalacOptions in Test ++= Seq("-Yrangepos", "-deprecation", "-feature", "-language:postfixOps", "-language:higherKinds", "-language:implicitConversions")