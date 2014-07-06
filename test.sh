#!/bin/bash

main="gpu/SoftmaxTest"
main="gpu/MinibatchTest"
main="gpu/RandTest"
main="MiscTest"
main="deep/SimpleSigmoidTest"

cp MyThrust/*.h bin/gpu/
cp matlab/*.txt bin/

cd bin

# an arbitrary command line arg tells the script to regenerate
if [ ! -z "$1" ]; then
java -jar "E:/Dropbox/Programming/Java/JarvisJava/BabelGPU/native/javacpp.jar" gpu/ThrustNative gpu/Thrust -properties windows-x86_64-cuda #> stderr.log
    
    rm -rf ../native/windows-x86_64
    cp -rf gpu/windows-x86_64 ../native/
else
    cp -rf ../native/windows-x86_64 ./gpu
fi

echo ;echo
echo 'Testing ...'; echo
java -Djava.library.path="E:/Dropbox/Programming/Java/Libraries/JCuda" -cp ".;E:/Dropbox/Programming/Java/Libraries/JCuda/jcuda-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcublas-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcurand-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar" test/$main
