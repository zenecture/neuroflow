libraryDependencies  ++= Seq(
  "org.specs2" %% "specs2-core" % "3.8.9" % "test",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

//scalacOptions in ThisBuild ++= Seq("-Xlog-implicits")