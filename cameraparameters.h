#pragma once

#include "camera.h"
#include "managed.h"
#include "config.h"

class __align__(128) CameraParameters_cu : public Managed {
public:
    float f; // used only for dummy depth -> disparity conversion
    bool rectified;
    Camera_cu cameras[MAX_IMAGES];
    int idRef;
    int cols;
    int rows;
    int* viewSelectionSubset;
    int viewSelectionSubsetNumber;
    CameraParameters_cu()
    {
		rectified = true;
		idRef = 0;
        cudaMallocManaged (&viewSelectionSubset, sizeof(int) * MAX_IMAGES);
    }
    ~CameraParameters_cu()
    {
        cudaFree (viewSelectionSubset);
    }
};
