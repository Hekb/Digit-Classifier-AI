# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.15.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.15.3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Matt/Desktop/code-2/task3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Matt/Desktop/code-2/task3

# Include any dependencies generated for this target.
include CMakeFiles/task3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/task3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/task3.dir/flags.make

CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.o: CMakeFiles/task3.dir/flags.make
CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.o: SimpleFeedForwardNetwork.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Matt/Desktop/code-2/task3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.o -c /Users/Matt/Desktop/code-2/task3/SimpleFeedForwardNetwork.cpp

CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Matt/Desktop/code-2/task3/SimpleFeedForwardNetwork.cpp > CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.i

CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Matt/Desktop/code-2/task3/SimpleFeedForwardNetwork.cpp -o CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.s

CMakeFiles/task3.dir/main.cpp.o: CMakeFiles/task3.dir/flags.make
CMakeFiles/task3.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Matt/Desktop/code-2/task3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/task3.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/task3.dir/main.cpp.o -c /Users/Matt/Desktop/code-2/task3/main.cpp

CMakeFiles/task3.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task3.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Matt/Desktop/code-2/task3/main.cpp > CMakeFiles/task3.dir/main.cpp.i

CMakeFiles/task3.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task3.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Matt/Desktop/code-2/task3/main.cpp -o CMakeFiles/task3.dir/main.cpp.s

# Object files for target task3
task3_OBJECTS = \
"CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.o" \
"CMakeFiles/task3.dir/main.cpp.o"

# External object files for target task3
task3_EXTERNAL_OBJECTS =

task3: CMakeFiles/task3.dir/SimpleFeedForwardNetwork.cpp.o
task3: CMakeFiles/task3.dir/main.cpp.o
task3: CMakeFiles/task3.dir/build.make
task3: CMakeFiles/task3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Matt/Desktop/code-2/task3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable task3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/task3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/task3.dir/build: task3

.PHONY : CMakeFiles/task3.dir/build

CMakeFiles/task3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/task3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/task3.dir/clean

CMakeFiles/task3.dir/depend:
	cd /Users/Matt/Desktop/code-2/task3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Matt/Desktop/code-2/task3 /Users/Matt/Desktop/code-2/task3 /Users/Matt/Desktop/code-2/task3 /Users/Matt/Desktop/code-2/task3 /Users/Matt/Desktop/code-2/task3/CMakeFiles/task3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/task3.dir/depend
