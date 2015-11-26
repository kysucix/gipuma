things that needed to be considered when creating the VS 2012 project (Windows):

windows.h in gipuma.cu (otherwise OpenGL Error)

direct.h, ctime.h in main.cpp (for mkdir)

opencv includes for applyColorMap in displayUtils.h
#include <opencv2/contrib.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>

at least for VS 2012 (maybe not any more for VS 2013) not all C++11 functions are already supported
eg. in class variable declaration (int xyz = 0;)
had to be changed in algorithmparameters.h, camera.h and cameraparameters.h

uint8_t not known (in gipuma.cu) --> include <stdint.h>

globalstate - use pointers!

Libs: (armadillo), OpenCV


Error vc110.pdb not loaded or so - no idea how i fixed it, probably properties config was wrong/missing something

Error Linker Error runcuda - return type of runcuda differed in .h and .cu



CUDA:

Parameters:
possibly need to set -rdc=true for dynamic parallelism (in CUDA C/C++ >> Common)
Code Generation: at least compute_35,sm_35 maybe even compute_52,sm_52 (in CUDA C/C++ >> Device)
verbose PTXAS Output (to see printf from device) (in CUDA C/C++ >> Device)
