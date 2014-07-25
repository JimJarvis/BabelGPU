package utils;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.Comparator;
import utils.PP;

/**
 * Data type that holds two components
 */
public class Pair<A, B> implements Comparable<Pair>, Serializable
{
    private static final long serialVersionUID = 1L;

	public A o1;
    public B o2;

    public Pair(A o1, B o2)
    {
    	this.o1 = o1;
    	this.o2 = o2;
    }

    /**
     * Default ctor
     */
    public Pair()
    {
    	this(null, null);
    }

    public String toString()
    {
           return "Pair(" + PP.o2str(o1) + ", " + PP.o2str(o2) + ")";
    }

    public void set1(A o1)
    { this.o1 = o1; }

    public void set2(B o2)
    { this.o2 = o2; }

    /**
     * Compares only the first element, given that it's comparable
     */
	public int compareTo(Pair that)
	{
		return ((Comparable) this.o1).compareTo(that.o1);
	}
	
	/**
	 * Python zip
	 * Must explicitly give class info. Ugly java generics. 
	 */
	public static <A, B> Pair<A[], B[]> zip(Pair<A, B>[] pairs, Class ca, Class cb)
	{
		int N = pairs.length;
		// Ugly workaround for Java's ugly generic erasure
		// hunt for the first A that isn't null
		A[] a = (A[]) Array.newInstance(ca, N);
		B[] b = (B[]) Array.newInstance(cb, N);
		for (int i = 0; i < N; ++i)
		{
			a[i] = pairs[i].o1;
			b[i] = pairs[i].o2;
		}
		return new Pair<>(a, b);
	}
	
	/**
	 * Python zip. 
	 * Ugly workaround, deduct class type from a concrete element. 
	 */
	public static <A, B> Pair<A[], B[]> zip(Pair<A, B>[] pairs)
	{
		int N = pairs.length;
		if (N == 0)
			throw new RuntimeException("Can't zip empty array");
		int i = -1;
		Class ca = null, cb = null;
		while (++ i < N)
			if (pairs[i].o1 != null)
			{
				ca = pairs[i].o1.getClass();
				break;
			}
		if (ca == null)
			throw new RuntimeException(
					"Can't zip: all instances of the first element are null");
		i = -1;
		while (++ i < N)
			if (pairs[i].o2 != null)
			{
				cb = pairs[i].o2.getClass();
				break;
			}
		if (cb == null)
			throw new RuntimeException(
					"Can't zip: all instances of the second element are null");
		return zip(pairs, ca, cb);
	}
	
	/**
	 * Python unzip
	 */
	public static <A, B> Pair<A, B>[] unzip(Pair<A[], B[]> pairArr)
	{
		int N = pairArr.o1.length;
		if (N != pairArr.o2.length)
			throw new RuntimeException("Can't unzip arrays of different lengths.");
		
		// Ugly workaround for Java ugly generics
		Pair<A, B> p = new Pair<A, B>(null, null);
		Pair<A, B> pairs[] = (Pair<A, B>[]) Array.newInstance(p.getClass(), N);
		for (int i = 0; i < N; ++i)
			pairs[i] = new Pair<>(pairArr.o1[i], pairArr.o2[i]);
		return pairs;
	}

	/**
	 * Treats the i-th element as Comparable. i starts at 1 <br>
	 * Default comparator: first and ascending
	 */
	public static Comparator<Pair> comparator(final int i, final boolean ascending)
	{
		return new Comparator<Pair>()
		{
			public int compare(Pair p1, Pair p2)
			{
				return (ascending ? 1 : -1) *
							(i == 1 ? ((Comparable)p1.o1).compareTo(p2.o1) :
						  				((Comparable)p1.o2).compareTo(p2.o2));
			}
		};
	}
	public static Comparator<Pair> comparator() { return comparator(1, true); }
	
	// Autogenerated
	public int hashCode()
	{
		final int prime = 31;
		int result = 1;
		result = prime * result + ((o1 == null) ? 0 : o1.hashCode());
		result = prime * result + ((o2 == null) ? 0 : o2.hashCode());
		return result;
	}

	// Autogenerated
	public boolean equals(Object obj)
	{
		if (this == obj) return true;
		if (obj == null) return false;
		if (getClass() != obj.getClass()) return false;
		Pair other = (Pair) obj;
		if (o1 == null)
		{
			if (other.o1 != null) return false;
		}
		else if (!o1.equals(other.o1)) return false;
		if (o2 == null)
		{
			if (other.o2 != null) return false;
		}
		else if (!o2.equals(other.o2)) return false;
		return true;
	}
}
