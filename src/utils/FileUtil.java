package utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import static java.nio.file.FileVisitResult.*;
import static java.nio.file.StandardCopyOption.*;
import java.util.ArrayList;
import java.util.Iterator;

public class FileUtil
{
	/**
	 * Handles access permissions. Use with '&' operator
	 */
	public static final int READ = 4;
	public static final int WRITE = 2;
	public static final int EXE = 1;
	
	/**
	 * Reads a file line by line, works in for-each loop
	 */
	public static Iterable<String> iterable(String file, String... file_)
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

	public static boolean isDir(String dir)
	{
		return Files.isDirectory(path(dir));
	}
	
	public static boolean exists(String file, String... file_)
	{
		return Files.exists(path(file, file_));
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
	 * List the contents of a directory. Doesn't include itself
	 * If not a dir, return empty list.
	 * @param deep whether or not we traverse the dir recursively. Default false
	 */
	public static ArrayList<Path> listDir(String dir, final boolean deep)
	{
		final ArrayList<Path> list = new ArrayList<>(); 
		if (!isDir(dir)) return list;
		try
		{
			Files.walkFileTree(Paths.get(dir), new FileVisitor<Path>()
				{
				boolean isRootDir = true;
				@Override
				public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException
				{
					if (!isRootDir) list.add(dir);
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
					list.add(file);
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
					return CONTINUE;
				}
			});
		}
		catch (IOException e) { e.printStackTrace(); }
		return list;
	}
	
	/**
	 * Recursive traversal default to false
	 * @see #listDir(String, boolean) listDir(String, false)
	 */
	public static ArrayList<Path> listDir(String dir) {	 return listDir(dir, false);	}
}
