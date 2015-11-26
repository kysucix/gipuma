#pragma once
#include <string.h> // memset()
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "managed.h"
#include <vector_types.h> // float4

class __align__(128) LineState : public Managed {
public:
    float4 *norm4; // 3 values for normal and last for d
    float *c; // cost
    /*float *disp; // disparity*/
    int n;
    int s; // stride
    int l; // length
    void resize(int n)
    {
        cudaMallocManaged (&c,        sizeof(float) * n);
        /*cudaMallocManaged (&disp,     sizeof(float) * n);*/
        cudaMallocManaged (&norm4,    sizeof(float4) * n);
        memset            (c,      0, sizeof(float) * n);
        /*memset            (disp,   0, sizeof(float) * n);*/
        memset            (norm4,  0, sizeof(float4) * n);
    }
    ~LineState()
    {
        cudaFree (c);
        cudaFree (norm4);
    }
};
