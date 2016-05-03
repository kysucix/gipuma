#pragma once
#include <vector_types.h> // float4
static __device__ float4 operator*(float4 a, float4 b) {
    return make_float4(a.x*b.x,
                       a.y*b.y,
                       a.z*b.z,
                       0);
}
static __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x-b.x,
                       a.y-b.y,
                       a.z-b.z,
                       0);
}
static __device__ float4 operator-(float4 a) {
    return make_float4(-a.x,
                       -a.y,
                       -a.z,
                       0);
}
static __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x+b.x,
                       a.y+b.y,
                       a.z+b.z,
                       0);
}
static __device__ float4 operator/(float4 a, float k) {
    return make_float4(a.x/k,
                       a.y/k,
                       a.z/k,
                       0);
}
static __device__ float l1_float4 (float4 a) {
    return ( fabsf (a.x) +
             fabsf (a.y) +
             fabsf (a.z))*0.3333333f;

}
