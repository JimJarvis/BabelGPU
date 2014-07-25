package utils;

import java.io.*;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;

import static java.nio.file.FileVisitResult.*;
import static java.nio.file.StandardCopyOption.*;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * Arg idiom: (String file, String... file_) means join the sub-paths. 'file_' is optional. 
 * Ex: ('rootdir', 'curdir', 'myfile') => 'rootdir/curdir/myfile'
 */
public class FileUtil
{
	/**
	 * Handles access permissions. Use with '&' operator
	 */
	public static final int READ = 4;
	public static final int WRITE = 2;
	public static final int EXE = 1;
	
	/**
	 * @return file path
	 * @see Paths#get(String, String...)
	 */
	public static Path path(String file, String... file_)	
	{
		// Shorthand for FileSystems.getDefault().getPath()
		return Paths.get(file, file_);
	}
	
	/**
	 * @return joined path
	 */
	public static String join(String file, String... file_)
	{
		return path(file, file_).toString();
	}
	
	/**
	 * Mimics python pickle
	 * Dump an object to a specified file
	 */
	public static <T extends Serializable> void dump(T obj, String file, String... file_)
	{
		ObjectOutputStream oos = null;
		try
		{
			oos = new ObjectOutputStream(new FileOutputStream(join(file, file_)));
			oos.writeObject(obj);
		}
		catch (IOException e) { e.printStackTrace(); }
		finally {	quietClose(oos);  }
	}

	/**
	 * Mimics python pickle 
	 * Load an object from a specified file
	 */
	public static <T extends Serializable> T load(String file, String... file_)
	{
        ObjectInputStream ois = null;
		try
		{
			ois = new ObjectInputStream(new FileInputStream(join(file, file_)));
			T obj = (T) ois.readObject();
			return obj;
		}
		catch (Exception e) { e.printStackTrace(); return null; }
		finally {	quietClose(ois);  }
	}
	
	/**
	 * Close a resource without throwing exception explicitly
	 */
	public static void quietClose(Closeable stream)
	{
		try { stream.close(); }
		catch (IOException e) { e.printStackTrace(); }
	}
	
	/**
	 * Reads a file line by line, works in for-each loop
	 */
	public static Iterable<String> lineIter(String file, String... file_)
	{
		final String filePath = join(file, file_);
		return new Iterable<String>()
		{
			@Override
			public Iterator<String> iterator()
			{
				return new Iterator<String>()
				{
    				BufferedReader reader;
    				String line;
    				// instance ctor
    				{
    					try {
    						reader = new BufferedReader(new FileReader(new File(filePath)));
    					}
    					catch (FileNotFoundException e) {	e.printStackTrace();}
    				}
					@Override
					public boolean hasNext()
					{
						try {
							return (line = reader.readLine()) != null;
						}
						catch (IOException e) { return false; }
					}

					@Override
					public String next() { return line; }

					@Override
					public void remove()
					{
						throw new UnsupportedOperationException();
					}
				};
			}
		};
	}
	
	/**
	 * Delete a file. If it's a directory, delete it recursively. 
	 * @param silentIfNotExist true to throw no exception when a file doesn't exist. 
	 * default true
	 */
	public static void delete(final boolean silentIfNotExist, String file, String... file_)
	{
		Path p = path(file, file_);
		try
		{
			if (! Files.isDirectory(p))
			{
				if (silentIfNotExist)
					Files.deleteIfExists(p);
				else
					Files.delete(p);
			}
			else
				Files.walkFileTree(p, new FileVisitor<Path>()
				{
					@Override
					public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException
					{
						return CONTINUE;
					}

					@Override
					public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException
					{
						Files.delete(file);
						return CONTINUE;
					}

					@Override
					public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException
					{
						return CONTINUE;
					}

					@Override
					public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException
					{
						if (exc == null)
						{
							Files.delete(dir);
							return CONTINUE;
						}
						else
							throw exc;
					}
				});
		}
		catch (IOException e) { e.printStackTrace(); }
	}
	
	/**
	 * Delete a file. If it's a directory, delete it recursively. 
	 * default silentIfNotExist = true
	 * @see #delete(String, String, boolean) delete(dir, fileName, true)
	 */
	public static void delete(String file, String... file_) {	delete(true, file, file_);	}
	
	/**
	 * @see #READ, WRITE, EXE
	 * @return access num to be '&' with READ ... constants
	 */
	public static int access(String file, String... file_)
	{
		Path p = path(file, file_);
		return 
				(Files.isReadable(p) ? READ : 0) | 
				(Files.isWritable(p) ? WRITE : 0) |
				(Files.isExecutable(p) ? EXE : 0);
	}

	public static boolean isDir(String dir, String... dir_)
	{
		return Files.isDirectory(path(dir, dir_));
	}
	
	public static boolean exists(String file, String... file_)
	{
		return Files.exists(path(file, file_));
	}
	
	public static String makeDir(String dir, String... dir_)
	{
		try {
			return Files.createDirectories(path(dir, dir_)).toString();
		}
		catch (IOException e) { e.printStackTrace(); return null; }
	}

	/**
	 * Make temporary directory
	 * @return temporary dir path
	 */
	public static String makeTempDir(String prefix, String dir, String... dir_)
	{
		try {
			return Files.createTempDirectory(path(dir, dir_), prefix).toString();
		}
		catch (IOException e) { e.printStackTrace(); return null; }
	}
	
	/**
	 * Make temporary file
	 * @return temporary file path
	 */
	public static String makeTempFile(String prefix, String suffix, String dir, String... dir_)
	{
		try {
			return Files.createTempFile(path(dir, dir_), prefix, suffix).toString();
		}
		catch (IOException e) { e.printStackTrace(); return null; }
	}
	
	/**
	 * Recursive copying. 
	 * If both source and target are files, copy and rename. 
	 * If source is file and target is folder, copy file into target's *inside*. 
	 * If both folder, copy the whole source folder *under* target folder. 
	 * If source is folder and target folder doesn't exist, create it and copy source folder to be new folder. 
	 */
	public static void copy(String source, String target)
	{
		final Path from = path(source);
		Path to_tmp = path(target);
		final Path to = Files.isDirectory(to_tmp) ? 
				to_tmp.resolve(from.getFileName()) : to_tmp;
		try
		{
			if (! Files.isDirectory(from))
				Files.copy(from, to, COPY_ATTRIBUTES, REPLACE_EXISTING);
			else
				Files.walkFileTree(from, new FileVisitor<Path>()
				{
					@Override
					public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException
					{
						Path newdir = to.resolve(from.relativize(dir));
						try {
							Files.copy(dir, newdir, COPY_ATTRIBUTES);
						} catch (FileAlreadyExistsException x) {
							// ignore
						} catch (IOException x) {
							System.err.format("Unable to create: %s: %s%n", newdir, x);
							return SKIP_SUBTREE;
						}
						return CONTINUE;
					}
					@Override
					public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException
					{
						Files.copy(file, to.resolve(from.relativize(file)), 
								COPY_ATTRIBUTES, REPLACE_EXISTING);
			            return CONTINUE;
					}
					@Override
					public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException
					{
						if (exc instanceof FileSystemLoopException) {
			                System.err.println("cycle detected: " + file);
			            } else {
			                System.err.format("Unable to copy: %s: %s%n", file, exc);
			            }
			            return CONTINUE;
					}
					@Override
					public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException
					{
						return CONTINUE;
					}
				});
		}
		catch (IOException e) { e.printStackTrace(); }
	}
	
	/**
	 * Recursive moving. 
	 * If both source and target are files, move and rename. 
	 * If source is file and target is folder, move file into target's *inside*. 
	 * If both folder, move the whole source folder *under* target folder. 
	 * If source is folder and target folder doesn't exist, create it and move source folder to be new folder. 
	 */
	public static void move(String source, String target)
	{
		final Path from = path(source);
		Path to_tmp = path(target);
		final Path to = Files.isDirectory(to_tmp) ? 
				to_tmp.resolve(from.getFileName()) : to_tmp;
		try
		{
			if (! Files.isDirectory(from))
				Files.move(from, to, REPLACE_EXISTING);
			else
				Files.walkFileTree(from, new FileVisitor<Path>()
				{
					@Override
					public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException
					{
						Path newdir = to.resolve(from.relativize(dir));
						try {
							Files.createDirectories(newdir);
						} catch (FileAlreadyExistsException x) {
							// ignore
						} catch (IOException x) {
							System.err.format("Unable to create: %s: %s%n", newdir, x);
							return SKIP_SUBTREE;
						}
						return CONTINUE;
					}
					@Override
					public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException
					{
						Files.move(file, to.resolve(from.relativize(file)), REPLACE_EXISTING);
			            return CONTINUE;
					}
					@Override
					public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException
					{
						if (exc instanceof FileSystemLoopException) {
			                System.err.println("cycle detected: " + file);
			            } else {
			                System.err.format("Unable to move: %s: %s%n", file, exc);
			            }
			            return CONTINUE;
					}
					@Override
					public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException
					{
						Files.delete(dir);
						return CONTINUE;
					}
				});
		}
		catch (IOException e) { e.printStackTrace(); }
	}
	
	/**
	 * Always returns true, matches everything
	 */
	public static final PathMatcher DummyMatcher = 
			new PathMatcher()
			{
				@Override
				public boolean matches(Path path) { return true; }
			};
			
	/**
	 * @param dirOnly true matches only directories, false matches only non-dirs
	 */
	public static PathMatcher dirMatcher(final boolean dirOnly)
	{
		return new PathMatcher()
		{
			@Override
			public boolean matches(Path path)
			{
				return Files.isDirectory(path) == dirOnly;
			}
		};
	}

	/**
	 * @param pattern if null or emptry string, matches everything 
	 * NOTE: we replace all '*' with '**' because only double * works across dir boundaries. 
	 * For ex, "*.txt" matches "dud.txt" but NOT "mydir/dud.txt"
	 * @return Pattern matcher for a glob string
	 */
	public static PathMatcher globMatcher(String pattern)
	{
		if (pattern == null || pattern.isEmpty())
			return DummyMatcher;

		if (!pattern.contains("**") && !pattern.contains("\\*") && pattern.contains("*"))
			pattern = pattern.replaceAll("\\*", "\\*\\*");
		return FileSystems.getDefault().getPathMatcher("glob:" + pattern);
	}

	/**
	 * List the contents of a directory w.r.t. pattern. Doesn't include itself
	 * If not a dir, return empty list. {@link http://docs.oracle.com/javase/tutorial/essential/io/find.html}
	 * @param matcher glob pattern matcher. null to use 'DummyPathMatcher' that matches everything
	 * @param deep whether or not we traverse the dir recursively. Default false
	 */
	public static ArrayList<String> listDir(String dir, final PathMatcher pathMatcher, final boolean deep)
	{
		final ArrayList<String> list = new ArrayList<>(); 
		if (!isDir(dir)) return list;
		try
		{
			Files.walkFileTree(Paths.get(dir), new FileVisitor<Path>()
				{
				boolean isRootDir = true;
				PathMatcher matcher = pathMatcher == null ? DummyMatcher : pathMatcher;
				
				@Override
				public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException
				{
					if (!isRootDir && matcher.matches(dir)) list.add(dir.toString());
					if (deep || isRootDir)
					{
						isRootDir = false;
						return CONTINUE;
					}
					else
						return SKIP_SUBTREE;
				}

				@Override
				public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException
				{
					if (matcher.matches(file)) list.add(file.toString());
					return CONTINUE;
				}

				@Override
				public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException
				{
					if (exc instanceof FileSystemLoopException) {
		                System.err.println("cycle detected: " + file);
		            }
					return CONTINUE;
				}

				@Override
				public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException
				{
					return CONTINUE;
				}
			});
		}
		catch (IOException e) { e.printStackTrace(); }
		return list;
	}
	
	/**
	 * List the contents of a directory w.r.t. pattern. Doesn't include itself
	 * @param matcher glob string pattern. 'null' or empty string to match everything
	 * @param deep whether or not we traverse the dir recursively. Default false
	 * @see #listDir(String, PathMatcher, boolean)
	 */
	public static ArrayList<String> listDir(String dir, String pattern, boolean deep)
	{
		return listDir(dir, globMatcher(pattern), deep);
	}
	
	/**
	 * Recursive traversal default to false. Matches everything
	 * @see #listDir(String, String, boolean) listDir(dir, null, false)
	 */
	public static ArrayList<String> listDir(String dir) {	 return listDir(dir, DummyMatcher, false);	}
	
	/**
	 * Writes to a file
	 */
	public static class Writer
	{
		private BufferedWriter writer;
		
		/**
		 * @param append false to overwrite an existing file
		 */
		public Writer(boolean append, String file, String... file_)
		{
			try {
				writer = new BufferedWriter(
						new FileWriter(join(file, file_), append));
			}
			catch (IOException e) { e.printStackTrace(); }
		}
		/**
		 * Default append = false
		 */
		public Writer(String file, String ... file_)
		{
			this(false, file, file_);
		}
		
		/**
		 * Write without new line
		 */
		public void write(Object ... objs)
		{
			try { writer.write(PP.all2str(objs)); }
			catch (IOException e) { e.printStackTrace(); }
		}
		
		/**
		 * Write with new line
		 */
		public void writeln(Object ... objs)
		{
			try { writer.write(PP.all2str(objs) + "\n"); }
			catch (IOException e) { e.printStackTrace(); }
		}

		/**
		 * Write a single object without newline
		 */
		public void write_(Object objs)
		{
			try { writer.write(PP.o2str(objs)); }
			catch (IOException e) { e.printStackTrace(); }
		}
		
		/**
		 * Write a single object with new line
		 */
		public void writeln_(Object objs)
		{
			try { writer.write(PP.o2str(objs) + "\n"); }
			catch (IOException e) { e.printStackTrace(); }
		}
		
		public void close() {	quietClose(writer);	}
	}
}
