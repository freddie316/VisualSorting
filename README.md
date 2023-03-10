# Visual Sorting
 Program for visualizing various sorting algorithms

https://user-images.githubusercontent.com/5875044/208347920-2ab771b2-5e99-426c-8e32-3c13e1e83b49.mp4
 
 ### Requirements:
 - Python 3.9  
 	- numpy
 	- scipy
 	- matplotlib
 - FFmpeg - https://github.com/BtbN/FFmpeg-Builds/releases
	- Required to compile the generated frames and sound into a mp4
	- Must be in system PATH so that .bat file can call it
	- Installation guide: https://www.youtube.com/watch?v=qjtmgCb8NcE&t=0s
 
 ### How To:
 Down the zip of the repository, unzip, and double click the make_movie.bat. Input what sorting method you want to use and the array size, then sit back and watch as the sorting algorithm runs!
 
 ### Available Methods (Recommended array size):
 - Bubble (N <= 50)
 - Cocktail (N <= 50)
 - Selection (N <= 50)
 - Insertion (N <= 50)
 - Shell (N <= 50)
 - Quick (N <= 100)
 - Merge (N <= 100)
 - Radix (N <= 100)

### Credits:  
Dave's Space - Awesome tutorial videos on creating this visualization method, https://github.com/davesspace/tutorial-visual-sorting
