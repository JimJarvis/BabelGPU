#!/bin/bash

main="test/ThrustTest"
#main="demo/JavacppDemo"

cp *.h bin/gpu/

cd bin
java -jar "E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar" gpu/Thrust gpu/ThrustStruct $main -properties windows-x86_64-cuda > stderr.log

#java -jar "E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar" $main -properties windows-x86_64-cuda > stderr.log

echo ;echo ;echo
echo 'Testing ...'; echo
java -Djava.library.path="E:/Dropbox/Programming/Java/Libraries/JCuda" -cp ".;E:/Dropbox/Programming/Java/Libraries/JCuda/jcuda-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcublas-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcurand-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar" $main
