package demo;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;

@Platform(include={"<algorithm>"}) 
@Namespace("std") 
public class JavacppDemo
{
	     static { Loader.load(); } 

	     public static class CompareInt extends FunctionPointer { 
	         static { Loader.load(); } 
	         public    CompareInt(Pointer p) { super(p); } 
	         protected CompareInt() { allocate(); } 
	         protected final native void allocate(); 
	         public native boolean call(int i1, int i2); 
	     } 
	     public static native void sort(IntPointer first, IntPointer last, @ByRef CompareInt compareInt); 

	     public static void main(String[] args) { 
	         CompareInt compareInt = new CompareInt() { 
	             public boolean call(int i1, int i2) { 
	                 return i1 < i2; 
	             } 
	         }; 
	         IntPointer first = new IntPointer(1024); 
	         IntPointer last = new IntPointer(first).position(1024); 
	         sort(first, last, compareInt); 
	     } 
}
