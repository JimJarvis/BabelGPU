#!/bin/bash
# to recompile the native libraries, ./test.sh c
# To do JUnit testing, ./test.sh blabla  (whatever string not 'c')

main="gpu/RandTest"
main="gpu/MinibatchTest"
main="deep/IniterTest"
main="gpu/MiscTest"

units="test.gpu.SoftmaxTest"
units="test.deep.ElementComputeTest test.deep.LinearTest test.deep.CombinedTest"


cp gpu_src/*.h bin/gpu/
cp matlab/test/*.txt bin/matlab_test/

cd bin

# an arbitrary command line arg tells the script to regenerate
if [[ $# -ne 0 && $1 == "c" ]]; then
java -jar "E:/Dropbox/Programming/Java/JarvisJava/BabelGPU/native/javacpp.jar" gpu/Natives gpu/Thrust -properties windows-x86_64-cuda #> stderr.log
    
    rm -rf ../native/windows-x86_64
    cp -rf gpu/windows-x86_64 ../native/
else
    cp -rf ../native/windows-x86_64 ./gpu
fi

echo ;echo
echo 'Testing ...'; echo
if [[ $# -eq 0 ]]; then
java -Djava.library.path="E:/Dropbox/Programming/Java/Libraries/JCuda" -cp ".;E:/Dropbox/Programming/Java/Libraries/JCuda/jcuda-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcublas-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcurand-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar" test/$main
else
java -Djava.library.path="E:/Dropbox/Programming/Java/Libraries/JCuda" -cp ".;E:/Dropbox/Programming/Java/Libraries/JCuda/jcuda-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcublas-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JCuda/jcurand-0.6.0.jar;E:/Dropbox/Programming/Java/Libraries/JavaCpp/javacpp.jar;E:/Dropbox/Programming/Java/Libraries/JUnit/junit-4.jar;E:/Dropbox/Programming/Java/Libraries/JUnit/hamcrest.jar" org.junit.runner.JUnitCore $units
fi
