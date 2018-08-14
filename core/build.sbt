val jcudaVersion = "0.8.0"

libraryDependencies  ++= Seq(
  "com.github.fommil.netlib"      %   "all"                            %  "1.1.2" pomOnly(),
  "org.scalanlp"                  %%  "breeze"                         %  "0.13.2",
  "org.scalanlp"                  %%  "breeze-natives"                 %  "0.13.2",
  "org.scalanlp"                  %%  "breeze-macros"                  %  "0.13.2" % "compile",
  "com.twitter"                   %   "chill_2.12"                     %  "0.9.2",
  "com.twitter"                   %   "chill-akka_2.12"                %  "0.9.2",
  "joda-time"                     %   "joda-time"                      %  "2.8.2",
  "org.slf4j"                     %   "slf4j-simple"                   %  "1.7.5",
  "org.specs2"                    %%  "specs2-core"                    %  "3.8.9" % "test",
  "org.scalatest"                 %%  "scalatest"                      %  "3.0.1" % "test",
  "com.nativelibs4java"           %   "bridj"                          %  "0.7.0",
  "com.zenecture"                 %   "jcuda"                          %  jcudaVersion,
  "com.zenecture"                 %   "jcuda-natives-apple-x86_64"     %  jcudaVersion,
  "com.zenecture"                 %   "jcuda-natives-windows-x86_64"   %  jcudaVersion,
  "com.zenecture"                 %   "jcuda-natives-linux-x86_64"     %  jcudaVersion,
  "com.zenecture"                 %   "jcublas"                        %  jcudaVersion,
  "com.zenecture"                 %   "jcublas-natives-apple-x86_64"   %  jcudaVersion,
  "com.zenecture"                 %   "jcublas-natives-windows-x86_64" %  jcudaVersion,
  "com.zenecture"                 %   "jcublas-natives-linux-x86_64"   %  jcudaVersion,
  "com.zenecture"                 %   "jcurand"                        %  jcudaVersion,
  "com.zenecture"                 %   "jcurand-natives-apple-x86_64"   %  jcudaVersion,
  "com.zenecture"                 %   "jcurand-natives-windows-x86_64" %  jcudaVersion,
  "com.zenecture"                 %   "jcurand-natives-linux-x86_64"   %  jcudaVersion
)

resolvers ++= Seq(
  Resolver.sonatypeRepo("snapshots"),
  Resolver.sonatypeRepo("releases"),
  "neuroflow-libs" at "https://github.com/zenecture/neuroflow-libs/raw/master/"
)
