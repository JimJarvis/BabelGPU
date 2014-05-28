#!/bin/bash

#main="test/MiscTest"
main="test/BabelTest"

cp MyThrust/*.h bin/gpu/
cp matlab/*.txt bin/

cd bin

# Before running test.sh, make sure to compile code in Eclipse
# If you change anything in ThrustNative or ThrustStruct, then you must recompile c++ code
# --- To do this, add a random string when calling ./test.sh  (eg, "./test.sh apple")

# an arbitrary command line arg tells the script to regenerate the compiled c++ header files (puts these files under bin/gpu/windows-x86_64)
if [ ! -z "$1" ]; then
	java -jar "E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar" gpu/ThrustNative gpu/ThrustStruct $main -properties windows-x86_64-cuda > stderr.log

	# removes the old version of the compiled c++ code (BabelGPU/windows-x86_64), and copies the latest version to this directory.
    rm -rf ../windows-x86_64
    cp -rf gpu/windows-x86_64 ../
else
    cp -rf ../windows-x86_64 ./gpu
fi

echo ;echo
echo 'Testing ...'; echo

# E:/Dropbox/Programming/Java/Libraries/JCuda: JCuda Dll's under here 
# Need to provide paths to jcuda*.jar, jcublas*.jar, jcurand*.jar, and javacpp.jar as classpaths.
java -Djava.library.path="E:/Dropbox/Programming/Java/Libraries/JCuda" -cp ".;E:/Dropbox/Programming/Java/Libraries/JCuda/jcuda-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcublas-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcurand-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar" $main
