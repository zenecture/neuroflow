libraryDependencies  ++= Seq(
  "org.specs2" %% "specs2-core" % "3.6.4" % "test",
  "org.scalatest" % "scalatest_2.11" % "2.2.6" % "test",
  "org.scala-lang.modules" %% "scala-pickling" % "0.10.1"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

//scalacOptions in ThisBuild ++= Seq("-Xlog-implicits")