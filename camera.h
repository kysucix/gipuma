#pragma once
#include "managed.h"
#include <vector_types.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

class Camera_cu : public Managed {
   public:
    float* P;
    float4 P_col34;
    float* P_inv;
    float* M_inv;
    float* R;
    float* R_orig_inv;
    float4 t4;
    float4 C4;
    float fx;
    float fy;
    float f;
    float alpha;
    float baseline;
    bool reference;
    float depthMin;  // this could be figured out from the bounding volume (not
                     // done right now, but that's why this parameter is here as
                     // well and not only in AlgorithmParameters)
    float depthMax;  // this could be figured out from the bounding volume (not
                     // done right now, but that's why this parameter is here as
                     // well and not only in AlgorithmParameters)
    char* id;  // corresponds to the image name id (eg. 0-10), independent of
               // order in argument list, just dependent on name
    float* K;
    float* K_inv;
    Camera_cu() {
        baseline = 0.54f;
        reference = false;
        depthMin = 2.0f;  // this could be figured out from the bounding volume
                          // (not done right now, but that's why this parameter
                          // is here as well and not only in
                          // AlgorithmParameters)
        depthMax = 20.0f;  // this could be figured out from the bounding volume
                           // (not done right now, but that's why this parameter
                           // is here as well and not only in
                           // AlgorithmParameters)

        checkCudaErrors(cudaMallocManaged(&P, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&P_inv, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&M_inv, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&K, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&K_inv, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&R, sizeof(float) * 4 * 4));
        checkCudaErrors(cudaMallocManaged(&R_orig_inv, sizeof(float) * 4 * 4));
    }
    ~Camera_cu() {
        cudaFree(P);
        cudaFree(P_inv);
        cudaFree(M_inv);
        cudaFree(K);
        cudaFree(K_inv);
        cudaFree(R);
        cudaFree(R_orig_inv);
    }
};
