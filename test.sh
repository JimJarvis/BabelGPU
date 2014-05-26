#!/bin/bash

#main="test/MiscTest"
main="test/BabelTest"

cp MyThrust/*.h bin/gpu/
cp dummy/*.txt bin/

cd bin

# an arbitrary command line arg tells the script to regenerate
if [ ! -z "$1" ]; then
java -jar "E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar" gpu/ThrustNative gpu/ThrustStruct $main -properties windows-x86_64-cuda > stderr.log
    
    rm -rf ../windows-x86_64
    cp -rf gpu/windows-x86_64 ../
else
    cp -rf ../windows-x86_64 ./gpu
fi

echo ;echo ;echo
echo 'Testing ...'; echo
java -Djava.library.path="E:/Dropbox/Programming/Java/Libraries/JCuda" -cp ".;E:/Dropbox/Programming/Java/Libraries/JCuda/jcuda-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcublas-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcurand-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar" $main
