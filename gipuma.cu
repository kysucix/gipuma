//#include <helper_math.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdint.h> // for uint8_t
#include "globalstate.h"
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "linestate.h"
#include "imageinfo.h"
#include "config.h"

#include <vector_types.h> // float4
#include <math.h>
#include <cuda.h>
#include <vector>
#include <string>
#include <iostream>
#include <curand_kernel.h>
#include "vector_operations.h"
#include "helper_cuda.h"


//#define CENSUS
#define SHARED
//#define NOTEXTURE_CHECK
#define WIN_INCREMENT 2

// uses smaller (but more) kernels, use if windows watchdog is enabled or if you want frequent display updates
// (might avoid crashes due to timeout on windows but possibly slows down marginally)
// only implements EXTRAPOINTFAR (not EXTRAPOINT or EXTRAPOINT2)
#define SMALLKERNEL


#define EXTRAPOINTFAR
#define EXTRAPOINT
#define EXTRAPOINT2

//#define FORCEINLINE __forceinline__
//#define FORCEINLINE


__device__ float K[16];
__device__ float K_inv[16];

#ifndef SHARED_HARDCODED
__managed__ int SHARED_SIZE_W_m;
__constant__ int SHARED_SIZE_W;
__managed__ int SHARED_SIZE_H;
__managed__ int SHARED_SIZE = 0;
__managed__ int WIN_RADIUS_W;
__managed__ int WIN_RADIUS_H;
__managed__ int TILE_W;
__managed__ int TILE_H;
#endif

/*__device__ FORCEINLINE __constant__ float4 camerasK[32];*/

/* compute depth value from disparity or disparity value from depth
 * Input:  f         - focal length in pixel
 *         baseline  - baseline between cameras (in meters)
 *         d - either disparity or depth value
 * Output: either depth or disparity value
 */
__device__ FORCEINLINE float disparityDepthConversion_cu ( const float &f, const float &baseline, const float &d ) {
    return f * baseline / d;
}

// CHECKED
__device__ FORCEINLINE void get3Dpoint_cu ( float4 * __restrict__ ptX, const Camera_cu &cam, const int2 &p, const float &depth ) {
    // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first
    const float4 pt = make_float4 (
                                   depth * (float)p.x     - cam.P_col34.x,
                                   depth * (float)p.y     - cam.P_col34.y,
                                   depth                  - cam.P_col34.z,
                                   0);

    matvecmul4 (cam.M_inv, pt, ptX);
}
__device__ FORCEINLINE void get3Dpoint_cu1 ( float4 * __restrict__ ptX, const Camera_cu &cam, const int2 &p) {
    // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first
    float4 pt;
    pt.x = (float)p.x     - cam.P_col34.x;
    pt.y = (float)p.y     - cam.P_col34.y;
    pt.z = 1.0f           - cam.P_col34.z;

    matvecmul4 (cam.M_inv, pt, ptX);
}
// CHECKED
//get d parameter of plane pi = [nT, d]T, which is the distance of the plane to the camera center
__device__ FORCEINLINE float getPlaneDistance_cu ( const float4 &normal, const float4 &X ) {
    return -(dot4(normal,X));
}
// CHECKED
__device__ FORCEINLINE static float getD_cu ( const float4 &normal,
                                              const int2 &p,
                                              const float &depth,
                                              const Camera_cu &cam ) {
    /*float4 pt;*/
    /*get3Dpoint_cu ( &pt, cam, (float)x0, (float)y0, depth );*/
    float4 pt,ptX;
    pt.x = depth * (float)(p.x)     - cam.P_col34.x;
    pt.y = depth * (float)(p.y)     - cam.P_col34.y;
    pt.z = depth         - cam.P_col34.z;

    matvecmul4 (cam.M_inv, pt, (&ptX));

    return -(dot4(normal,ptX));
    /*return getPlaneDistance_cu (normal, ptX);*/
}
// CHECKED
__device__ FORCEINLINE void normalize_cu (float4 * __restrict__ v)
{
    const float normSquared = pow2(v->x) + pow2(v->y) + pow2(v->z);
    const float inverse_sqrt = rsqrtf (normSquared);
    v->x *= inverse_sqrt;
    v->y *= inverse_sqrt;
    v->z *= inverse_sqrt;
}
//CHECKED
__device__ FORCEINLINE void getViewVector_cu (float4 * __restrict__ v, const Camera_cu &camera, const int2 &p)
{
    get3Dpoint_cu1 (v, camera, p);
    sub((*v), camera.C4);
    normalize_cu(v);
    //v->x=0;
    //v->y=0;
    //v->z=1;
}
__device__ FORCEINLINE static void vecOnHemisphere_cu ( float4 * __restrict__ v, const float4 &viewVector ) {
    const float dp = dot4 ( (*v), viewVector );
    if ( dp > 0.0f ) {
        negate4(v);
    }
    return;
}
__device__ FORCEINLINE float curand_between (curandState *cs, const float &min, const float &max)
{
    return (curand_uniform(cs) * (max-min) + min);
}
/* compute random 3D unit vector
 * notes on how to do that: http://mathworld.wolfram.com/SpherePointPicking.html
 * this method uses the last approach in the link by Muller 1959, Marsaglia 1972: three Gaussian random values for x,y,z
 * Output: random unit vector
 */
// CHECKED
__device__ FORCEINLINE static void rndUnitVectorSphereMarsaglia_cu (float4 *v, curandState *cs) {
    float x = 1.0f;
    float y = 1.0f;
    float sum = 2.0f;
    while ( sum>=1.0f ) {
        x = curand_between (cs, -1.0f, 1.0f);
        y = curand_between (cs, -1.0f, 1.0f);
        sum = get_pow2_norm(x,y);
    }
    const float sq = sqrtf ( 1.0f-sum );
    v->x = 2.0f*x*sq;
    v->y = 2.0f*y*sq;
    v->z = 1.0f-2.0f*sum;
    //v->x = 0;
    //v->y = 0;
    //v->z = -1;
}
//CHECKED
__device__ FORCEINLINE static void rndUnitVectorOnHemisphere_cu ( float4 *v, const float4 &viewVector, curandState *cs ) {
    rndUnitVectorSphereMarsaglia_cu (v, cs);
    vecOnHemisphere_cu ( v,viewVector );
};

__device__ FORCEINLINE float l1_norm(float f) {
    return fabsf(f);
}
__device__ FORCEINLINE float l1_norm(float4 f) {
    return ( fabsf (f.x) +
             fabsf (f.y) +
             fabsf (f.z))*0.3333333f;

}
__device__ FORCEINLINE float l1_norm2(float4 f) {
    return ( fabsf (f.x) +
             fabsf (f.y) +
             fabsf (f.z));

}
template< typename T >
__device__ FORCEINLINE float weight_cu ( const T &c1, const T &c2, const float &gamma )
{
    const float colorDis = l1_norm (  c1 -  c2 );
    return expf ( -colorDis / gamma ); ///[>0.33333333f));
    /*return __expf ( -colorDis / gamma ); ///[>0.33333333f));*/
    /*return expf_cache[c1-c2+256];*/
}

// CHECKED
__device__ FORCEINLINE void getCorrespondingHomographyPt_cu ( const float * __restrict__ H, int x, int y, float4 *ptf) {
    float4 pt;
    pt.x = __int2float_rn (x);
    pt.y = __int2float_rn (y);
    pt.z = 1.0f;
    matvecmul4(H,pt,ptf); //ptf =  H * pt;
    vecdiv4(ptf,ptf->z); //ptf = ptf / ptf[2];

    return ;
}
// CHECKED
__device__ FORCEINLINE void getCorrespondingPoint_cu ( const int2 &p, const float * __restrict__ H, float4 * __restrict__ ptf ) {
    /*getCorrespondingHomographyPt_cu ( (const float * )H, x , y , pt );*/
    float4 pt;
    pt.x = __int2float_rn (p.x);
    pt.y = __int2float_rn (p.y);
    /*pt.z = 1.0f;*/
    matvecmul4noz(H,pt,ptf); //ptf =  H * pt;
    vecdiv4(ptf,ptf->z); //ptf = ptf / ptf[2];

    return ;
}
__device__ FORCEINLINE float colorDifferenceL1_cu ( float c1, float c2 )
{
    return abs ( c1-c2 );
}

template< typename T >
__device__ FORCEINLINE float pmCostComputation_shared (
                                                       const cudaTextureObject_t &l,
                                                       const T * __restrict__ tile_left,
                                                       const cudaTextureObject_t &r,
                                                       const T &leftValue,
                                                       const int2 &pI,
                                                       const float4 &pt_r,
                                                       const float &tau_color,
                                                       const float &tau_gradient,
                                                       const float &alpha,
                                                       const float &w )
{
    /*XXX*/
    /*if ( pt_r.x >= 0 && */
    /*pt_r.x < cols && */
    /*pt_r.y >= 0 && */
    /*pt_r.y < rows ) */
    {
        /*float dis = dissimilarity ( l, r, pt_l, pt_r, gradX1, gradY1, gradX2, gradY2, alpha, tau_color, tau_gradient );*/

        /*float colDiff = colorDifferenceL1_cu ( texatpt4(l,pt_l), texatpt4(r,pt_r) );*/

        /*if (*/
        /*pt_li.x == 100 && */
        /*pt_li.y == 100)*/
        /*printf ("PMCOSTCOMPUTATION I and J are %d %d and II JJ are %d %d value is %f and tile cache is %f\n", I, J, II, JJ, leftValue, tile_left[I][J]);*/

        const T gx2   = tex2D<T> (r, pt_r.x+1 + 0.5f, pt_r.y   + 0.5f) - tex2D<T> (r, pt_r.x-1 + 0.5f, pt_r.y   + 0.5f);
        const T gy2   = tex2D<T> (r, pt_r.x   + 0.5f, pt_r.y+1 + 0.5f) - tex2D<T> (r, pt_r.x   + 0.5f, pt_r.y-1 + 0.5f);
        const float colDiff = l1_norm ( leftValue - tex2D<T>(r, pt_r.x + 0.5f, pt_r.y + 0.5f) );
        const T up    = tile_left[ pI.x   + SHARED_SIZE_W * (pI.y-1)];
        const T down  = tile_left[ pI.x   + SHARED_SIZE_W * (pI.y+1)];
        const T left  = tile_left[ pI.x-1 + SHARED_SIZE_W *  pI.y  ];
        const T right = tile_left[ pI.x+1 + SHARED_SIZE_W *  pI.y  ];
        const T gx1   = right - left;
        const T gy1   = down - up;

        /*float gradX = texatpt4(gradX1,pt_l) - texatpt4(gradX2,pt_r);*/
        /*float gradY = texatpt4(gradY1,pt_l) - texatpt4(gradY2,pt_r);*/
        const T gradX = (gx1 - gx2);
        const T gradY = (gy1 - gy2);

        //gradient dissimilarity (L1) in x and y direction (multiplication by 0.5 to use tauGrad from PatchMatch stereo paper)
        const float gradDis = fminf ( ( l1_norm ( gradX ) + l1_norm ( gradY ) ) * 0.0625f, tau_gradient );
        //gradient dissimilarity only in x direction
        //float gradDis = min(abs(gradX),tau_gradient);

        const float colDis = fminf ( colDiff, tau_color );
        const float dis = ( 1.f - alpha ) * colDis + alpha * gradDis;
        //const float dis = gradDis;
        return w * dis;
    }
    //return 3.0;
}
template< typename T >
__device__ FORCEINLINE float pmCostComputation (
                                                const cudaTextureObject_t &l,
                                                const T * __restrict__ tile_left,
                                                const cudaTextureObject_t &r,
                                                const float4 &pt_l,
                                                const float4 &pt_r,
                                                const int &rows,
                                                const int &cols,
                                                const float &tau_color,
                                                const float &tau_gradient,
                                                const float &alpha,
                                                const float &w )
{
    /*XXX*/
    /*if ( pt_r.x >= 0 && */
    /*pt_r.x < cols && */
    /*pt_r.y >= 0 && */
    /*pt_r.y < rows ) */
    {
        /*float dis = dissimilarity ( l, r, pt_l, pt_r, gradX1, gradY1, gradX2, gradY2, alpha, tau_color, tau_gradient );*/

        const float colDiff = l1_norm ( tex2D<T>(l,pt_l.x + 0.5f,pt_l.y + 0.5f) - tex2D<T>(r,pt_r.x + 0.5f, pt_r.y + 0.5f) );
        const float colDis = fminf ( colDiff, tau_color );

        const T gx1 = tex2D<T> (l, pt_l.x+1 + 0.5f, pt_l.y   + 0.5f) - tex2D<T> (l, pt_l.x-1 + 0.5f, pt_l.y   + 0.5f);
        const T gy1 = tex2D<T> (l, pt_l.x   + 0.5f, pt_l.y+1 + 0.5f) - tex2D<T> (l, pt_l.x   + 0.5f, pt_l.y-1 + 0.5f);
        const T gx2 = tex2D<T> (r, pt_r.x+1 + 0.5f, pt_r.y   + 0.5f) - tex2D<T> (r, pt_r.x-1 + 0.5f, pt_r.y   + 0.5f);
        const T gy2 = tex2D<T> (r, pt_r.x   + 0.5f, pt_r.y+1 + 0.5f) - tex2D<T> (r, pt_r.x   + 0.5f, pt_r.y-1 + 0.5f);

        const T gradX = (gx1 - gx2);
        const T gradY = (gy1 - gy2);

        //gradient dissimilarity (L1) in x and y direction (multiplication by 0.5 to use tauGrad from PatchMatch stereo paper)
        const float gradDis = fminf ( ( l1_norm ( gradX ) + l1_norm ( gradY ) ) * 0.0625f, tau_gradient );
        //gradient dissimilarity only in x direction
        //float gradDis = min(abs(gradX),tau_gradient);

        const float dis = ( 1.f - alpha ) * colDis + alpha * gradDis;
        return w * dis;
    }
    //return 3.0;
}
__device__ FORCEINLINE void getHomography_real (const float *K1_inv,
                                                const float *K2,
                                                const float *R,
                                                const float4 t,
                                                const float4 n,
                                                const float d,
                                                float *H )
{
    /*print_matrix(R,"R");*/
    float tmp[16];
    float tmp2[16];
    outer_product4(t, n, tmp); // tmp = t * n'
    matdivide(tmp, d); // tmp / d
    matmatsub2(R, tmp); // tmp = R - tmp;
    matmul_cu(tmp,K1_inv,tmp2); // tmp2=tmp*Kinv
    matmul_cu(K2,tmp2,H);// H = tmp * K2
    return;
}
__device__ FORCEINLINE void getHomography_cu ( const Camera_cu &from, const Camera_cu &to,
                                               const float * __restrict__ K1_inv, const float * __restrict__ K2,
                                               const float4 &n, const float &d, float * __restrict__ H )
{
    //if ( !to.reference )
    {
        /*getHomography_real( K1_inv, K2, to.R, to.t4, n, d, H );*/
        /*float tmp[16];*/
        float tmp2[16];
        outer_product4(to.t4, n, H); // tmp = t * n'
        matdivide(H, d); // tmp / d
        matmatsub2(to.R, H); // tmp = R - tmp;
        matmul_cu(H,K1_inv,tmp2); // tmp2=tmp*Kinv
        matmul_cu(K2,tmp2,H);// H = tmp * K2

    }
    return;
}
/* census transform, get value based on weather intensity is smaller or higher than center intensity
 * Input: p        - intensity of center pixel
 *        pNb      - intensity of current nb pixel in the kernel
 *        epsilon  - threshold for classifying as the same intensity (for original ct no epsilon is used --> epsilon=0)
 */
__device__ FORCEINLINE uint8_t getCTbit_cu ( float p, float pNb, float eps ) {
    uint8_t bit = 1;
    if ( p - pNb > eps )
        bit = 0;
    else if ( pNb - p > eps )
        bit = 2;
    return bit;
}
__device__ FORCEINLINE float ct_Arma_cu ( const cudaTextureObject_t &l,
                                          const cudaTextureObject_t &r,
                                          const int2 &p,
                                          const int vRad,
                                          const int hRad,
                                          const float intensityCenterLeft,
                                          const float intensityCenterRight,
                                          const float eps,
                                          const float* __restrict__ H
                                        )
{
    float4 pt;
    getCorrespondingPoint_cu ( p, H,  &pt );


    //default cost if pt outside of image
    float c = 1.0f;

    //if ( pt ( 0 ) > 0 &&
    //pt ( 0 ) < ( float ) ( l.cols - 1 ) &&
    //pt ( 1 ) > 0 &&
    //pt ( 1 ) < ( float ) ( l.rows - 1 ) )
    {
        float intensityL = texatpt4(l, p);
        float intensityR = texatpt4(r,pt);

        if ( getCTbit_cu ( intensityCenterLeft, intensityL, eps ) == getCTbit_cu ( intensityCenterRight, intensityR, eps ) )
            c = 0.0f;
        else
            c = 1.0f;
    }
    return c;
}
/* census transform cost computation for search window
 * check if intensity of neighboring pixel is lower or higher than intensity of center pixel
 */
__device__ float censusTransform_Arma_cu ( const cudaTextureObject_t &l,
                                           const cudaTextureObject_t &r,
                                           const int2 &p,
                                           const float &d,
                                           const int &vRad,
                                           const int &hRad,
                                           const float &eps,
                                           const float* __restrict__ H)
{
    float cost = 0.0f;
    float4 pt;
    getCorrespondingPoint_cu ( p, H,  &pt );

    /*if ( pt_c ( 0 ) <= 0.0f || */
    /*pt_c ( 0 ) >= ( float ) ( l.cols - 1 ) || */
    /*pt_c ( 1 ) <= 0.0f || */
    /*pt_c ( 1 ) >= ( float ) ( l.rows - 1 ) ||*/
    /*pt_c ( 0 ) != pt_c (0) ||*/
    /*pt_c ( 1 ) != pt_c (1) )*/
    /*{*/
    /*return ( float ) ( hRad * 2 + 1 ) * ( vRad * 2 + 1 );*/
    /*}*/

    float intensityCenterLeft  = texatpt4(l, p);
    float intensityCenterRight = texatpt4(r,pt);
    //if (blockIdx.x ==0 && blockIdx.y ==0) printf("color is %f %f\n", intensityCenterLeft, intensityCenterRight);

    //use non-border values for disparity computation
    //subtract i and j by half kernel size since disparity is without border
    for ( int i = p.x - hRad; i <= p.x + hRad; i++ ) {
        for ( int j = p.y - vRad; j <= p.y + vRad; j++ ) {

            if ( i == p.x && j == p.y )
                continue;

            float c = ct_Arma_cu ( l, r, make_int2(i, j), vRad, hRad, intensityCenterLeft, intensityCenterRight, eps, H);

            //w = weight_cu ( leftValue, centerValue, gamma);

            cost = cost + c;
        }
    }

    return cost;
}

/*
 * cost computation of different cost functions
 */
template< typename T >
__device__ FORCEINLINE static float pmCost (
                                            const cudaTextureObject_t &l,
                                            const T * __restrict__ tile_left,
                                            const int2 tile_offset,
                                            const cudaTextureObject_t &r,
                                            const int &x,
                                            const int &y,
                                            const float4 &normal,
                                            const int &vRad,
                                            const int &hRad,
                                            const AlgorithmParameters &algParam,
                                            const CameraParameters_cu &camParams,
                                            const int &camTo )
{
    const int cols = camParams.cols;
    const int rows = camParams.rows;
    const float alpha = algParam.alpha;
    const float tau_color = algParam.tau_color;
    const float tau_gradient = algParam.tau_gradient;
    const float gamma = algParam.gamma;

    float4 pt_c;
    float H[16];
    /*float H[3*3];*/
    getHomography_cu ( camParams.cameras[REFERENCE], camParams.cameras[camTo], camParams.cameras[REFERENCE].K_inv, camParams.cameras[camTo].K, normal, normal.w, H );
    getCorrespondingPoint_cu ( make_int2(x, y), H, &pt_c );

    // XXX to review
    //if (    pt_c.x  < hRad ||
    //pt_c.x  >= ( float ) ( cols - hRad - 1 ) ||
    //pt_c.y  < ( float ) vRad ||
    //pt_c.y  >= ( float ) ( rows - vRad - 1 ) ) {
    //return 1000; // XXX
    //}

    {
        float cost = 0;
        //float weightSum = 0.0f;
        for ( int i = -hRad; i < hRad + 1; i+=WIN_INCREMENT ) {
            for ( int j = -vRad; j < vRad + 1; j+=WIN_INCREMENT ) {
                const int xTemp = x + i;
                const int yTemp = y + j;
                float4 pt_l;
                pt_l.x = __int2float_rn(xTemp);
                pt_l.y = __int2float_rn(yTemp);
                int2 pt_li = make_int2(xTemp, yTemp);

                float w;

                w = weight_cu<T> ( tex2D<T>(l, pt_l.x + 0.5f, pt_l.y + 0.5f), tex2D<T>(l,x + 0.5f,y + 0.5f), gamma);

                float4 pt;
                getCorrespondingPoint_cu ( make_int2(xTemp, yTemp),
                                           H,
                                           &pt );

                cost = cost + pmCostComputation<T> ( l, tile_left, r, pt_l, pt, rows, cols, tau_color, tau_gradient, alpha,  w );
                //weightSum = weightSum + w;
            }
        }
        return cost;
    }
}

template< typename T >
__device__ FORCEINLINE static float hasImageTexture (
                                                   const cudaTextureObject_t &l,
                                                   const int2 &p,
                                                   const int &vRad,
                                                   const int &hRad,
                                                   const AlgorithmParameters &algParam)
{
    const float gamma = algParam.gamma;

    int count_similar_pixel = 0;
    for ( int i = -hRad; i < hRad + 1; i += WIN_INCREMENT ) {
        for ( int j = -vRad; j < vRad + 1; j += WIN_INCREMENT ) {
            const int xTemp = p.x + i;
            const int yTemp = p.y + j;
            float4 pt_l;
            pt_l.x = __int2float_rn( xTemp );
            pt_l.y = __int2float_rn( yTemp );

            const float w = weight_cu<T> ( tex2D <T> (l, pt_l.x + 0.5f, pt_l.y + 0.5f ), tex2D <T> ( l, p.x + 0.5f, p.y + 0.5f ), gamma);
                if (w > algParam.no_texture_sim)
                    count_similar_pixel++;
        }
    }
    if (count_similar_pixel > hRad*vRad*4/(WIN_INCREMENT * WIN_INCREMENT)*algParam.no_texture_per)
        return false;
    return true;
}

template< typename T >
__device__ FORCEINLINE static float hasImageTexture_shared (
                                                   const cudaTextureObject_t &l,
                                                   const T * __restrict__ tile_left,
                                                   const int2 tile_offset,
                                                   const int2 &p,
                                                   const int &vRad,
                                                   const int &hRad,
                                                   const AlgorithmParameters &algParam)
{
        const T centerValue = tile_left[ p.x-tile_offset.x + SHARED_SIZE_W * ( p.y - tile_offset.y ) ];

        int count_similar_pixel = 0;
        for ( int i = -hRad; i < hRad + 1; i += WIN_INCREMENT) {
            for ( int j = -vRad; j < vRad + 1; j += WIN_INCREMENT) {
                const int2 pI = make_int2 ( p.x + i - tile_offset.x, p.y + j - tile_offset.y);
                const T leftValue = tile_left[ pI.x + SHARED_SIZE_W * pI.y ];
                const float w = weight_cu<T> ( leftValue,
                                   centerValue,
                                   algParam.gamma);
                //if (p.x == 440 && p.y == 307 )
                    //printf("Weight is %f\tValues are %f and %f\n", w, centerValue, leftValue);

                if (w > algParam.no_texture_sim)
                    count_similar_pixel++;
            }
        }
        //if (p.x == 440 && p.y == 307 ) {
            //printf("Count similar pixel is %d\n", count_similar_pixel);
            ////printf("Limit is %f\n", (float) 4*hRad*vRad/(WIN_INCREMENT * WIN_INCREMENT)*algParam.no_texture_per);
            ////printf("Hrad is %d\n", vRad);
        //}
    if (count_similar_pixel > hRad*vRad*4/(WIN_INCREMENT * WIN_INCREMENT)*algParam.no_texture_per)
            return false;
        return true;
}
template< typename T >
__device__ FORCEINLINE static float pmCost_shared (
                                                   const cudaTextureObject_t &l,
                                                   const T * __restrict__ tile_left,
                                                   const int2 tile_offset,
                                                   const cudaTextureObject_t &r,
                                                   const int2 &p,
                                                   const float4 &normal,
                                                   const int &vRad,
                                                   const int &hRad,
                                                   const AlgorithmParameters &algParam,
                                                   const CameraParameters_cu &camParams,
                                                   const int &camTo )
{
    const float alpha = algParam.alpha;
    const float tau_color = algParam.tau_color;
    const float tau_gradient = algParam.tau_gradient;
    const float gamma = algParam.gamma;

    /*float4 pt_c;*/
    float H[16];
    /*float H[3*3];*/
    //getHomography_cu ( camParams.cameras[REFERENCE], camParams.cameras[camTo], camParams.K_inv, camParams.K, normal, normal.w, H );
    getHomography_cu ( camParams.cameras[REFERENCE], camParams.cameras[camTo], camParams.cameras[REFERENCE].K_inv, camParams.cameras[camTo].K, normal, normal.w, H );

    /*getCorrespondingPoint_cu ( x, y, H, &pt_c );*/

    // XXX to review
    //if (    pt_c.x  < hRad ||
    //pt_c.x  >= ( float ) ( cols - hRad - 1 ) ||
    //pt_c.y  < ( float ) vRad ||
    //pt_c.y  >= ( float ) ( rows - vRad - 1 ) ) {
    //return 1000; // XXX
    //}

    {
        float cost = 0;
        //float weightSum = 0.0f;

        //const int Ic = x - tile_offset.x;
        //const int Jc = y - tile_offset.y;
        const T centerValue = tile_left[p.x-tile_offset.x + SHARED_SIZE_W*(p.y-tile_offset.y)];

#ifdef CENSUS
        cost = censusTransform_Arma_cu (l, r, p, normal.w, vRad, hRad, algParam.census_epsilon, H);
        return cost;
#endif

        for ( int i = -hRad; i < hRad + 1; i+=WIN_INCREMENT) {
            for ( int j = -vRad; j < vRad + 1; j+=WIN_INCREMENT) {
                const int2 pTemp = make_int2(p.x +i, p.y + j);
                //const int xTemp = p.x + i;
                //const int yTemp = p.y + j;

                const int2 pI = make_int2 ( p.x + i - tile_offset.x, p.y + j - tile_offset.y);

                float w;
#if 0
                if (tile_offset.x !=0 &&
                    xTemp <12 &&
                    yTemp < 12
                   )
                {
                    if (texatpt4(l,pt_l) != tile_left[I+SHARED_SIZE_W*J])
                    {
                        //printf("PMCOST x %d %d Xtemp %d %d \t\tI %d J %d tilecoords %d %d offset is %d %d blockIdx %d %d tile_offset.x %d tile_offset.y %d\n", x, y, xTemp, yTemp, I, J, xTemp-tile_offset.x, yTemp-tile_offset.y, tile_offset.x, tile_offset.y, blockIdx.x, blockIdx.y, tile_offset.x, tile_offset.y);
                        printf("Tex is %f, caache is %f\nPMCOST x %d %d Xtemp %d %d \t\tI %d J %d tilecoords %d %d offset is %d %d blockIdx %d %d tile_offset.x %d tile_offset.y %d\n", texatpt4(l, pt_l), tile_left[I+SHARED_SIZE_W*J], x, y, xTemp, yTemp, I, J, xTemp-tile_offset.x, yTemp-tile_offset.y, tile_offset.x, tile_offset.y, blockIdx.x, blockIdx.y, tile_offset.x, tile_offset.y);
                    }
                }
#endif
                const T leftValue = tile_left[pI.x + SHARED_SIZE_W*pI.y];
                /*if (tile_offset.x !=0 && */
                /*xTemp == 100 && */
                /*yTemp == 100) {*/
                /*printf ("I and J are %d %d and value is %f\n", I, J,leftValue);*/
                /*}*/
                w = weight_cu<T> ( leftValue,
                                   centerValue,
                                   gamma);
                //if( p.x == 446 && p.y == 307)
                    //printf("weigth is %f\n", w);

                //const float w = weight_cu ( tile_left[xTemp - SHARED_SIZE_H][], pt_l), texat(l,x,y), gamma);

                float4 pt;
                getCorrespondingPoint_cu ( pTemp,
                                           H,
                                           &pt );

                cost = cost + pmCostComputation_shared<T> ( l, tile_left, r, leftValue, pI, pt, tau_color, tau_gradient, alpha, w );
                //weightSum = weightSum + w;
            }
        }
        return cost;
    }
}


// via https://stackoverflow.com/questions/2786899/fastest-sort-of-fixed-length-6-int-array
static __device__ FORCEINLINE void sort_small(float * __restrict__ d,const int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
            d[j] = d[j-1];
        d[j] = tmp;
    }
}
__device__ FORCEINLINE float getDepthFromPlane3_cu (const Camera_cu &cam,
                                                    const float4 &n,
                                                    const float &d,
                                                    const int2 &p)
{
    return -d*cam.fx/(
                      (n.x*(p.x-cam.K[2]))
                      +
                      (n.y*(p.y-cam.K[2+3]))
                      *cam.alpha +
                      n.z*cam.fx);
}
__device__ FORCEINLINE float getDisparity_cu ( const float4 &normal,
                                               const float &d,
                                               const int2 &p,
                                               const Camera_cu &cam )
{
    if ( d != d )
        return 1000;

    return getDepthFromPlane3_cu (cam, normal, d, p);
}

/* cost computation for multiple images
 * combines cost of all ref-to-img correspondences
 */
template< typename T >
__device__ FORCEINLINE static float pmCostMultiview_cu (
                                                        const cudaTextureObject_t *images,
                                                        const T * __restrict__ tile_left,
                                                        const int2 tile_offset,
                                                        const int2 p,
                                                        const float4 &normal,
                                                        const int &vRad,
                                                        const int &hRad,
                                                        const AlgorithmParameters &algParam,
                                                        const CameraParameters_cu &camParams,
                                                        const float4 * __restrict__ state,
                                                        const int point)
{
    // iterate over all other images and compute cost
    //const int numImages = camParams.viewSelectionSubsetNumber; // CACHE
    float costVector[32];
    float cost = 0.0f;
    int numValidViews = 0;


    int cost_count=0;
    for ( int i = 0; i < camParams.viewSelectionSubsetNumber; i++ ) {
        int idxCurr = camParams.viewSelectionSubset[i];
        /*if ( idxCurr != REFERENCE ) */
        {
            float c = 0;
#ifdef SHARED
            if (tile_offset.x!= 0 )
                c = pmCost_shared<T> ( images[REFERENCE],
                                       tile_left,
                                       tile_offset,
                                       images[idxCurr],
                                       p,
                                       normal,
                                       vRad, hRad,
                                       algParam, camParams,
                                       idxCurr );
            else
#endif
                c = pmCost<T> ( images[REFERENCE],
                                tile_left,
                                tile_offset,
                                images[idxCurr],
                                p.x, p.y,
                                normal,
                                vRad, hRad,
                                algParam, camParams,
                                idxCurr );

            // only add to cost vector if viewable
            if ( c < MAXCOST )
                numValidViews++;
            else
                c = MAXCOST; // in order to not get an overflow when accumulating
            costVector[i] = c;
            cost_count++;
        }
    }
    sort_small(costVector,cost_count);

    //for some robustness only consider best n cost values (n dependent on number of images)
    int numBest = numValidViews; //numImages-1;
    if ( algParam.cost_comb == COMB_BEST_N )
        numBest = min ( numBest, algParam.n_best );
    if ( algParam.cost_comb == COMB_GOOD )
        numBest = camParams.viewSelectionSubsetNumber ;

    float costThresh = costVector[0] * algParam.good_factor;
    int numConsidered = 0;
    for ( int i = 0; i < numBest; i++ ) {
        numConsidered++;
        float c = costVector[i];
        if ( algParam.cost_comb == COMB_GOOD ) {
            c = fminf ( c, costThresh );
        }
        cost = cost + c;
    }
    cost = cost / ( ( float ) numConsidered);
    if ( numConsidered < 1 )
        cost = MAXCOST;

    if ( cost != cost || cost > MAXCOST || cost < 0 )
        cost = MAXCOST;

    return cost;
}

__device__ FORCEINLINE float get_smoothness_at2 ( const float4 * __restrict__ state,
                                                  const float4 &norm,
                                                  const float &depth,
                                                  const int2 p,
                                                  const int2 p_other,
                                                  const int cols,
                                                  const Camera_cu &cam )
{
    float4 norm_other = state [p_other.x + p_other.y*cols];
    const float depth_other = getDisparity_cu (norm_other, norm_other.w, p_other, cam);

    float4 X_other;
    float4 X;
    get3Dpoint_cu (&X,       cam, p,       depth);
    get3Dpoint_cu (&X_other, cam, p_other, depth_other);

    return (1.0f - fabsf(dot4(norm,norm_other)) + 1.0f);
}



#define ISDISPDEPTHWITHINBORDERS(disp,camParams,camIdx,algParams) \
disp >= camParams.cameras[REFERENCE].depthMin && disp <= camParams.cameras[REFERENCE].depthMax

template< typename T >
__device__ FORCEINLINE void spatialPropagation_cu ( const cudaTextureObject_t *imgs,
                                                    const T * __restrict__ tile_left,
                                                    const int2 &tile_offset,
                                                    const int2 &p,
                                                    const int &box_hrad, const int &box_vrad,
                                                    const AlgorithmParameters &algParams,
                                                    const CameraParameters_cu &camParams,
                                                    float *cost_now,
                                                    float4 *norm_now,
                                                    const float4 norm_before,
                                                    float *disp_now,
                                                    const float4 * __restrict__ state,
                                                    const int point
                                                  )
{
    // previous image values

    const float d_before    = norm_before.w;
    const float disp_before = getDisparity_cu (norm_before, d_before, p, camParams.cameras[REFERENCE] );

    float cost_before = pmCostMultiview_cu<T> ( imgs,
                                                tile_left,
                                                tile_offset,
                                                p,
                                                norm_before,
                                                box_vrad,
                                                box_hrad,
                                                algParams,
                                                camParams,
                                                state,
                                                point);

    if ( ISDISPDEPTHWITHINBORDERS(disp_before,camParams,REFERENCE,algParams) )
    {
        if ( cost_before < *cost_now ) {
            *disp_now   = disp_before;
            *norm_now   = norm_before;
            *cost_now   = cost_before;
        }
    }
    return;
}

/* compute random disparity and unit vector within given intervals, used for plane refinement step
 * interval is limited by image border and general disparity range [0 maxDisparity]
 * Input: x     - current column x
 *        disp  - old disparity value
 *        norm  - old normal
 *        maxDeltaZ  - range radius for disparity [disp-maxDeltaZ,disp+maxDeltaZ]
 *        maxDeltaN  - range radius for normal
 *        maxDisparity  - maximum disparity value
 *        cols - number of columns of the image
 *        dir - disparity to the left or right of x
 *        limit - defines maximal value for |[nx ny]T| so that only plane tilts to a certain degree are possible
 * Output: dispOut - new disparity
 *         normOut - new normal
 */
__device__ FORCEINLINE void getRndDispAndUnitVector_cu (
                                                        float disp,
                                                        const float4 norm,
                                                        float &dispOut,
                                                        float4 * __restrict__ normOut,
                                                        const float maxDeltaZ,
                                                        const float maxDeltaN,
                                                        const float minDisparity,
                                                        const float maxDisparity,
                                                        curandState *cs,
                                                        CameraParameters_cu &camParams,
                                                        const float baseline,
                                                        const float4 viewVector) {
    //convert depth to disparity and back for non-rectified approach
    disp = disparityDepthConversion_cu ( camParams.f, baseline, disp );

    //delta min limited by disp=0 and image border
    //delta max limited by disp=maxDisparity and image border
    float minDelta, maxDelta;
    minDelta = -min ( maxDeltaZ, minDisparity + disp ); //limit new disp>=0
    maxDelta = min ( maxDeltaZ, maxDisparity - disp ); //limit new disp < maxDisparity

    /*minDelta ; -minDelta;*/

    float deltaZ = curand_between(cs, minDelta, maxDelta);
    //get new disparity value within valid range [0 maxDisparity]
    dispOut = fminf ( fmaxf ( disp + deltaZ, minDisparity ), maxDisparity );

    dispOut = disparityDepthConversion_cu ( camParams.f, baseline, dispOut );

    //get normal
    normOut->x = norm.x + curand_between (cs, -maxDeltaN, maxDeltaN );
    normOut->y = norm.y + curand_between (cs, -maxDeltaN, maxDeltaN );
    normOut->z = norm.z + curand_between (cs, -maxDeltaN, maxDeltaN );

    normalize_cu ( normOut );
    vecOnHemisphere_cu (  normOut, viewVector );
}
template< typename T >
__device__ FORCEINLINE static void planeRefinement_cu (
                                                       const cudaTextureObject_t *images,
                                                       const T * __restrict__ tile_left,
                                                       const int2 &p,
                                                       const int2 &tile_offset,
                                                       const int &box_hrad,
                                                       const int &box_vrad,
                                                       const AlgorithmParameters &algParams,
                                                       CameraParameters_cu &camParams,
                                                       const int camIdx,
                                                       float * __restrict__ cost_now,
                                                       float4 * __restrict__ norm_now,
                                                       float * __restrict__ disp_now,
                                                       curandState *cs,
                                                       const float4 * __restrict__ state)
{
    float deltaN = 1.0f;

    float4 viewVector;
    getViewVector_cu (&viewVector, camParams.cameras[0], p);

    // divide delta by 4 instead of 2 for less iterations (for higher disparity range)
    // iteration is done over disparity values even for multi-view case in order to have approximately unifom sampling along epipolar line
    /*for ( float deltaZ = ( float ) algParams.max_disparity / 2.0f; deltaZ >= 0.1f; deltaZ = deltaZ / 4.0f ) {*/
    float4 norm_temp;
    float dispTemp_L;
    float dTemp_L;
    float costTempL;

    const float maxdisp=algParams.max_disparity / 2.0f; // temp variable
   for ( float deltaZ = maxdisp; deltaZ >= 0.01f; deltaZ = deltaZ / 10.0f ) {
        getRndDispAndUnitVector_cu (
                                    *disp_now, *norm_now,
                                    dispTemp_L, &norm_temp,
                                    deltaZ, deltaN,
                                    algParams.min_disparity, algParams.max_disparity,
                                    cs,
                                    camParams, camParams.cameras[0].baseline,
                                    viewVector);

        dTemp_L = getD_cu ( norm_temp,
                            p,
                            dispTemp_L, camParams.cameras[camIdx] );

        norm_temp.w = dTemp_L; // TODO might save a variable here
        costTempL = pmCostMultiview_cu<T> ( images,
                                            tile_left,
                                            tile_offset,
                                            p,
                                            norm_temp,
                                            box_vrad, box_hrad,
                                            algParams, camParams,
                                            state,
                                            0);

        //if (dTemp_L==dTemp_L && dTemp_L!= 0) // XXX
        {
            if ( costTempL < *cost_now ) {
                *cost_now = costTempL;
                *disp_now = dispTemp_L;
                *norm_now = norm_temp;
            }
        }
        deltaN = deltaN / 4.0f;
    }
}

template< typename T >
__global__ void gipuma_init_cu2(GlobalState &gs)
{
    const int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;

    // Temporary variables
    Camera_cu &camera = gs.cameras->cameras[REFERENCE];

    const int center = p.y*cols+p.x;
    int box_hrad = gs.params->box_hsize / 2;
    int box_vrad = gs.params->box_vsize / 2;

    float disp_now;
    float4 norm_now;

    curandState localState = gs.cs[p.y*cols+p.x];
    curand_init ( clock64(), p.y, p.x, &localState );

    // Compute random normal on half hemisphere of fronto view vector
    float mind = gs.params->min_disparity;
    float maxd = gs.params->max_disparity;
    float4 viewVector;
    getViewVector_cu ( &viewVector, camera, p);
    //printf("Random number is %f\n", random_number);
    //return;
    disp_now = curand_between(&localState, mind, maxd);

    rndUnitVectorOnHemisphere_cu ( &norm_now, viewVector, &localState );
    disp_now= disparityDepthConversion_cu ( camera.f, camera.baseline, disp_now);

    // Save values
    norm_now.w = getD_cu ( norm_now, p, disp_now,  camera);
    //disp[x] = disp_now;
    gs.lines->norm4[center] = norm_now;

    __shared__ T tile_leftt[1] ;
    const int2 tmp =make_int2(0,0);
    gs.lines->c[center] = pmCostMultiview_cu<T> ( gs.imgs,
                                                 tile_leftt,
                                                 tmp,
                                                 p,
                                                 norm_now,
                                                 box_vrad, box_hrad,
                                                 *(gs.params),
                                                 *(gs.cameras),
                                                 gs.lines->norm4,
                                                 0);
    return;
}
template< typename T >
__global__ void gipuma_initial_cost(GlobalState &gs)
{
    const int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;

    const int center = p.y*cols+p.x;
    int box_hrad = gs.params->box_hsize / 2;
    int box_vrad = gs.params->box_vsize / 2;

    __shared__ T tile_leftt[1] ;
    const int2 tmp =make_int2(0,0);
    gs.lines->c[center] = pmCostMultiview_cu<T> ( gs.imgs,
                                                 tile_leftt,
                                                 tmp,
                                                 p,
                                                 gs.lines->norm4[center],
                                                 box_vrad, box_hrad,
                                                 *(gs.params), *(gs.cameras), gs.lines->norm4, 0);

    return;
}
__global__ void gipuma_compute_disp (GlobalState &gs)
{
    const int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;

    const int center = p.y*cols+p.x;
    float4 norm = gs.lines->norm4[center];
    float4 norm_transformed;
    // Transform back normal to world coordinate
    matvecmul4 ( gs.cameras->cameras[REFERENCE].R_orig_inv, norm, (&norm_transformed));
    //vecOnHemisphere_cu ( &norm, viewVector );
    if (gs.lines->c[center] != MAXCOST)
        norm_transformed.w = getDisparity_cu (norm, norm.w, p, gs.cameras->cameras[REFERENCE] );
    else
        norm_transformed.w = 0;
    gs.lines->norm4[center] = norm_transformed;
    return;
}

__global__ void gipuma_init_random (GlobalState &gs)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int2 p = make_int2(x,y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;
    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;

    curandState localState = gs.cs[y*cols+x];
    curand_init ( clock64(), y, x, &localState );
    return;
}
template< typename T >
__device__ FORCEINLINE void gipuma_checkerboard_cu(GlobalState &gs, int2 p, const int2 tile_offset, int iter)
{
    int box_hrad = (gs.params->box_hsize-1) / 2;
    int box_vrad = (gs.params->box_vsize-1) / 2;
    //if (iter > ITER/4) {
        //box_hrad *=2;
        //box_vrad *=2;
    //}
    //if (iter >= ITER-1) {
        //box_hrad *=2;
        //box_vrad *=2;
    //}

    /*printf("%d: First line is %f\n", idx, ls.d[idx]);*/

    const LineState &line = *(gs.lines);
    float *c     = line.c;
    //float *disp  = line.disp;
    float4 *norm = line.norm4;

    float disp_now;
    float cost_now;
    float4 norm_now;
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    curandState localState = gs.cs[p.y*cols+p.x];

    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;
    // Center
    const int center = p.y*cols+p.x;

    // Left
    const int left = center-1;
    const int leftleft = center-3;

    // Up
    const int up = center-cols;
    const int upup = center-3*cols;

    // Down
    const int down = center+cols;
    const int downdown = center+3*cols;

    // Right
    const int right = center+1;
    const int rightright = center+3;

    AlgorithmParameters &algParams = *(gs.params);
    CameraParameters_cu &camParams = *(gs.cameras);
    const cudaTextureObject_t *imgs = gs.imgs;

    /*float *tile_left_tmp;*/

    //extern __shared__ float tile_left[];
    // via https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T *tile_left = reinterpret_cast<T *>(my_smem);

#ifdef SHARED
    // Load shared_size_w * SHARED_SIZE_H into shared memory as a cache
    // In y direction be careful of the checkerboard pattern
    {
        int ii = 7*(threadIdx.y*TILE_W + threadIdx.x); // flattened indexing

        for (int t = 0 ; t < 8 ; t++) {
            const int I1 = ii%(SHARED_SIZE_W); // Index of shared area x
            const int J1 = ii/(SHARED_SIZE_H);
            const int x1 = blockIdx.x * TILE_W + I1 - WIN_RADIUS_W;
            const int y1 = blockIdx.y * TILE_H + J1 - WIN_RADIUS_H;
            if (ii<SHARED_SIZE)
                tile_left[ii++] = tex2D <T> (imgs[REFERENCE], x1 + 0.5f, y1 + 0.5f);
        }
    }

    __syncthreads();
#endif
    // Read from global memory
    cost_now = c       [center];
    norm_now = norm    [center];
    disp_now = getDisparity_cu (norm_now, norm_now.w, p, camParams.cameras[REFERENCE] );
    //norm_now.z = -sqrtf(1.0f - (norm_now.x*norm_now.x + norm_now.y*norm_now.y));

#ifdef NOTEXTURE_CHECK
    {
        bool hasTexture;
        // Check for non textured area and exclude it
#ifdef SHARED
        if (tile_offset.x!= 0 )
            hasTexture = hasImageTexture_shared<T> (imgs[REFERENCE],
                                                    tile_left,
                                                    tile_offset,
                                                    p,
                                                    box_vrad,
                                                    box_hrad,
                                                    algParams);
        else
#endif
            hasTexture = hasImageTexture<T> (imgs[REFERENCE],
                                             p,
                                             box_vrad,
                                             box_hrad,
                                             algParams);
        if (!hasTexture) {
            c    [center] = MAXCOST;
            return;
        }
    }
#endif

#define SPATIALPROPAGATION(point) spatialPropagation_cu<T> (imgs, tile_left, tile_offset, p, box_hrad, box_vrad, algParams, camParams, &cost_now, &norm_now, norm[point], &disp_now, norm, point)

    if (p.y>0) {
        SPATIALPROPAGATION(up);
    }
#ifdef EXTRAPOINT
    if (p.y>2) {
        SPATIALPROPAGATION(upup);
    }
#endif
#ifdef EXTRAPOINTFAR
    if (p.y>4) {
        SPATIALPROPAGATION(upup-cols*2);
    }
#endif

    if (p.y<rows-1) {
        SPATIALPROPAGATION(down);
    }
#ifdef EXTRAPOINT
    if (p.y<rows-3) {
        SPATIALPROPAGATION(downdown);
    }
#endif
#ifdef EXTRAPOINTFAR
    if (p.y<rows-5) {
        SPATIALPROPAGATION(downdown+cols*2);
    }
#endif

    if (p.x>0) {
        SPATIALPROPAGATION(left);
    }
#ifdef EXTRAPOINT
    if (p.x>2) {
        SPATIALPROPAGATION(leftleft);
    }
#endif
#ifdef EXTRAPOINTFAR
    if (p.x>4) {
        SPATIALPROPAGATION(leftleft-2);
    }
#endif

    if (p.x<cols-1) {
        SPATIALPROPAGATION(right);
    }
#ifdef EXTRAPOINT
    if (p.x<cols-3)  {
        SPATIALPROPAGATION(rightright);
    }
#endif
#ifdef EXTRAPOINTFAR
    if (p.x<cols-5) {
        SPATIALPROPAGATION(rightright+2);
    }
#endif

#ifdef EXTRAPOINT2
    if (p.y>0 &&
        p.x<cols-2) {
        SPATIALPROPAGATION(up+2);
    }
    if (p.y< rows-1 &&
        p.x<cols-2) {
        SPATIALPROPAGATION(down+2);
    }
    if (p.y>0 &&
        p.x>1)
    {
        SPATIALPROPAGATION(up-2);
    }
    if (p.y<rows-1 &&
        p.x>1) {
        SPATIALPROPAGATION(down-2);
    }
    if (p.x>0 &&
        p.y>2)
    {
        SPATIALPROPAGATION(left  - cols*2);
    }
    if (p.x<cols-1 &&
        p.y>2)
    {
        SPATIALPROPAGATION(right - cols*2);
    }
    if (p.x>0 &&
        p.y<rows-2) {
        SPATIALPROPAGATION(left  + cols*2);
    }
    if (p.x<cols-1 &&
        p.y<rows-2) {
        SPATIALPROPAGATION(right + cols*2);
    }
#endif

    planeRefinement_cu<T> (imgs,
                        tile_left,
                        p,
                        tile_offset,
                        box_hrad, box_vrad,
                        algParams, camParams,
                        REFERENCE,
                        &cost_now,
                        &norm_now,
                        &disp_now,
                        &localState,
                        norm);

    // Save to global memory
    c    [center] = cost_now;
    //disp [center] = disp_now;
    norm [center] = norm_now;

    return;
}

template< typename T >
__device__ FORCEINLINE void gipuma_checkerboard_spatialPropFar_cu(GlobalState &gs, int2 p, const int2 tile_offset, int iter)
{
    int box_hrad = (gs.params->box_hsize-1) / 2;
    int box_vrad = (gs.params->box_vsize-1) / 2;

    const LineState &line = *(gs.lines);
    float *c     = line.c;
    //float *disp  = line.disp;
    float4 *norm = line.norm4;

    float disp_now;
    float cost_now;
    float4 norm_now;
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;

    AlgorithmParameters &algParams = *(gs.params);
    CameraParameters_cu &camParams = *(gs.cameras);
    const cudaTextureObject_t *imgs = gs.imgs;

    /*float *tile_left_tmp;*/

    //extern __shared__ float tile_left[];
    // via https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T *tile_left = reinterpret_cast<T *>(my_smem);

     // Center
    const int center = p.y*cols+p.x;

    #ifdef SHARED
    // Load shared_size_w * SHARED_SIZE_H into shared memory as a cache
    // In y direction be careful of the checkerboard pattern
    {
        int ii = 7*(threadIdx.y*TILE_W + threadIdx.x); // flattened indexing

        for (int t = 0 ; t < 8 ; t++) {
            const int I1 = ii%(SHARED_SIZE_W); // Index of shared area x
            const int J1 = ii/(SHARED_SIZE_H);
            const int x1 = blockIdx.x * TILE_W + I1 - WIN_RADIUS_W;
            const int y1 = blockIdx.y * TILE_H + J1 - WIN_RADIUS_H;
            if (ii<SHARED_SIZE)
                tile_left[ii++] = tex2D <T> (imgs[REFERENCE], x1 + 0.5f, y1 + 0.5f);
        }
    }

    __syncthreads();
#endif
    // Read from global memory
    cost_now = c       [center];
    norm_now = norm    [center];
    disp_now = getDisparity_cu (norm_now, norm_now.w, p, camParams.cameras[REFERENCE] );

#ifdef NOTEXTURE_CHECK
    {
        bool hasTexture;
        // Check for non textured area and exclude it
#ifdef SHARED
        if (tile_offset.x!= 0 )
            hasTexture = hasImageTexture_shared<T> (imgs[REFERENCE],
                                                    tile_left,
                                                    tile_offset,
                                                    p,
                                                    box_vrad,
                                                    box_hrad,
                                                    algParams);
        else
#endif
            hasTexture = hasImageTexture<T> (imgs[REFERENCE],
                                             p,
                                             box_vrad,
                                             box_hrad,
                                             algParams);
        if (!hasTexture) {
            c    [center] = MAXCOST;
            return;
        }
    }
#endif

    // Left by 5
    const int left = center-5;
    // Up by 5
    const int up = center-5*cols;
    // Down by 5
    const int down = center+5*cols;
    // Right by 5
    const int right = center+5;

    #define SPATIALPROPAGATION(point) spatialPropagation_cu<T> (imgs, tile_left, tile_offset, p, box_hrad, box_vrad, algParams, camParams, &cost_now, &norm_now, norm[point], &disp_now, norm, point)

    if (p.y>4) {
        SPATIALPROPAGATION(up);
    }

    if (p.y<rows-5) {
        SPATIALPROPAGATION(down);
    }
    if (p.x>4) {
        SPATIALPROPAGATION(left);
    }
    if (p.x<cols-5) {
        SPATIALPROPAGATION(right);
    }

    // Save to global memory
    c    [center] = cost_now;
    //disp [center] = disp_now;
    norm [center] = norm_now;
}


template< typename T >
__device__ FORCEINLINE void gipuma_checkerboard_spatialPropClose_cu(GlobalState &gs, int2 p, const int2 tile_offset, int iter)
{
    int box_hrad = (gs.params->box_hsize-1) / 2;
    int box_vrad = (gs.params->box_vsize-1) / 2;

    const LineState &line = *(gs.lines);
    float *c     = line.c;
    //float *disp  = line.disp;
    float4 *norm = line.norm4;

    float disp_now;
    float cost_now;
    float4 norm_now;
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;

    AlgorithmParameters &algParams = *(gs.params);
    CameraParameters_cu &camParams = *(gs.cameras);
    const cudaTextureObject_t *imgs = gs.imgs;

    /*float *tile_left_tmp;*/

    //extern __shared__ float tile_left[];
    // via https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T *tile_left = reinterpret_cast<T *>(my_smem);

     // Center
    const int center = p.y*cols+p.x;

    #ifdef SHARED
    // Load shared_size_w * SHARED_SIZE_H into shared memory as a cache
    // In y direction be careful of the checkerboard pattern
    {
        /*const int i  = threadIdx.x;*/
        /*const int j  = threadIdx.y; // Thread indexes within block*/
        int ii = 7*(threadIdx.y*TILE_W + threadIdx.x); // flattened indexing

        for (int t = 0 ; t < 8 ; t++) {
            const int I1 = ii%(SHARED_SIZE_W); // Index of shared area x
            const int J1 = ii/(SHARED_SIZE_H);
            const int x1 = blockIdx.x * TILE_W + I1 - WIN_RADIUS_W;
            const int y1 = blockIdx.y * TILE_H + J1 - WIN_RADIUS_H;
            if (ii<SHARED_SIZE)
                tile_left[ii++] = tex2D <T> (imgs[REFERENCE], x1 + 0.5f, y1 + 0.5f);
        }
    }

    __syncthreads();
#endif
    // Read from global memory
    cost_now = c       [center];
    norm_now = norm    [center];
    disp_now = getDisparity_cu (norm_now, norm_now.w, p, camParams.cameras[REFERENCE] );
    //norm_now.z = -sqrtf(1.0f - (norm_now.x*norm_now.x + norm_now.y*norm_now.y));

#ifdef NOTEXTURE_CHECK
    {
        bool hasTexture;
        // Check for non textured area and exclude it
#ifdef SHARED
        if (tile_offset.x!= 0 )
            hasTexture = hasImageTexture_shared<T> (imgs[REFERENCE],
                                                    tile_left,
                                                    tile_offset,
                                                    p,
                                                    box_vrad,
                                                    box_hrad,
                                                    algParams);
        else
#endif
            hasTexture = hasImageTexture<T> (imgs[REFERENCE],
                                             p,
                                             box_vrad,
                                             box_hrad,
                                             algParams);
        if (!hasTexture) {
            c    [center] = MAXCOST;
            return;
        }
    }
#endif

    // Left
    const int left = center-1;
    // Up
    const int up = center-cols;
    // Down
    const int down = center+cols;
    // Right
    const int right = center+1;

    #define SPATIALPROPAGATION(point) spatialPropagation_cu<T> (imgs, tile_left, tile_offset, p, box_hrad, box_vrad, algParams, camParams, &cost_now, &norm_now, norm[point], &disp_now, norm, point)

    if (p.y>0) {
        SPATIALPROPAGATION(up);
    }
    if (p.y<rows-1) {
        SPATIALPROPAGATION(down);
    }
    if (p.x>0) {
        SPATIALPROPAGATION(left);
    }
    if (p.x<cols-1) {
        SPATIALPROPAGATION(right);
    }

    // Save to global memory
    c    [center] = cost_now;
    //disp [center] = disp_now;
    norm [center] = norm_now;
}

template< typename T >
__device__ FORCEINLINE void gipuma_checkerboard_planeRefinement_cu(GlobalState &gs, int2 p, const int2 tile_offset, int iter)
{
    int box_hrad = (gs.params->box_hsize-1) / 2;
    int box_vrad = (gs.params->box_vsize-1) / 2;

    const LineState &line = *(gs.lines);
    float *c     = line.c;
    //float *disp  = line.disp;
    float4 *norm = line.norm4;

    float disp_now;
    float cost_now;
    float4 norm_now;
    const int rows = gs.cameras->rows;
    const int cols = gs.cameras->cols;

    // TODO move in a separate kernel
    curandState localState = gs.cs[p.y*cols+p.x];
    //curand_init ( clock64(), y, x, &localState );

    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;
    // Center
    const int center = p.y*cols+p.x;


    AlgorithmParameters &algParams = *(gs.params);
    CameraParameters_cu &camParams = *(gs.cameras);
    const cudaTextureObject_t *imgs = gs.imgs;

    /*float *tile_left_tmp;*/

    //extern __shared__ float tile_left[];
    // via https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T *tile_left = reinterpret_cast<T *>(my_smem);

    //__shared__ T tile_left [SHARED_SIZE];
    //__shared__ float tile_right [SHARED_SIZE];
    //float* tile_left  = (float *)  shared_memory;
    //float* tile_right = (float *)  &shared_memory[SHARED_SIZE];
    //float4* norm_tile = (float4 *) &shared_memory[SHARED_SIZE];
    //float* d_tile     = (float *)  &shared_memory[SHARED_SIZE_CHECKER];

#ifdef SHARED
    // Load shared_size_w * SHARED_SIZE_H into shared memory as a cache
    // In y direction be careful of the checkerboard pattern
    {
        /*const int i  = threadIdx.x;*/
        /*const int j  = threadIdx.y; // Thread indexes within block*/
        int ii = 7*(threadIdx.y*TILE_W + threadIdx.x); // flattened indexing

        for (int t = 0 ; t < 8 ; t++) {
            const int I1 = ii%(SHARED_SIZE_W); // Index of shared area x
            const int J1 = ii/(SHARED_SIZE_H);
            const int x1 = blockIdx.x * TILE_W + I1 - WIN_RADIUS_W;
            const int y1 = blockIdx.y * TILE_H + J1 - WIN_RADIUS_H;
            if (ii<SHARED_SIZE)
                tile_left[ii++] = tex2D <T> (imgs[REFERENCE], x1 + 0.5f, y1 + 0.5f);
        }
    }

    __syncthreads();
#endif
    // Read from global memory
    cost_now = c       [center];
    norm_now = norm    [center];
    disp_now = getDisparity_cu (norm_now, norm_now.w, p, camParams.cameras[REFERENCE] );
    //norm_now.z = -sqrtf(1.0f - (norm_now.x*norm_now.x + norm_now.y*norm_now.y));


#ifdef NOTEXTURE_CHECK
    {
        bool hasTexture;
        // Check for non textured area and exclude it
#ifdef SHARED
        if (tile_offset.x!= 0 )
            hasTexture = hasImageTexture_shared<T> (imgs[REFERENCE],
                                                    tile_left,
                                                    tile_offset,
                                                    p,
                                                    box_vrad,
                                                    box_hrad,
                                                    algParams);
        else
#endif
            hasTexture = hasImageTexture<T> (imgs[REFERENCE],
                                             p,
                                             box_vrad,
                                             box_hrad,
                                             algParams);
        if (!hasTexture) {
            c    [center] = MAXCOST;
            return;
        }
    }
#endif


    planeRefinement_cu<T> (imgs,
                        tile_left,
                        p,
                        tile_offset,
                        box_hrad, box_vrad,
                        algParams, camParams,
                        REFERENCE,
                        &cost_now,
                        &norm_now,
                        &disp_now,
                        &localState,
                        norm);

    // Save to global memory
    c    [center] = cost_now;
    //disp [center] = disp_now;
    norm [center] = norm_now;

    return;
}

template< typename T >
__global__ void gipuma_black_cu(GlobalState &gs, int iter)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2;
    else
        p.y = p.y*2 + 1;
    int2 tile_offset;
    tile_offset.x =       blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    gipuma_checkerboard_cu<T>(gs, p, tile_offset, iter);
}

template< typename T >
__global__ void gipuma_black_spatialPropClose_cu(GlobalState &gs, int iter)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2;
    else
        p.y = p.y*2 + 1;
    int2 tile_offset;
    tile_offset.x =       blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    gipuma_checkerboard_spatialPropClose_cu<T>(gs, p, tile_offset, iter);
}

template< typename T >
__global__ void gipuma_black_spatialPropFar_cu(GlobalState &gs, int iter)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2;
    else
        p.y = p.y*2 + 1;
    int2 tile_offset;
    tile_offset.x =       blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    gipuma_checkerboard_spatialPropFar_cu<T>(gs, p, tile_offset, iter);
}

template< typename T >
__global__ void gipuma_black_planeRefine_cu(GlobalState &gs, int iter)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2;
    else
        p.y = p.y*2 + 1;
    int2 tile_offset;
    tile_offset.x =       blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    gipuma_checkerboard_planeRefinement_cu<T>(gs, p, tile_offset, iter);
}

template< typename T >
__global__ void gipuma_red_cu(GlobalState &gs, int iter)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2 + 1;
    else
        p.y = p.y*2;
    int2 tile_offset;
    tile_offset.x =       blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    gipuma_checkerboard_cu<T>(gs, p, tile_offset, iter);
}

template< typename T >
__global__ void gipuma_red_spatialPropClose_cu(GlobalState &gs, int iter)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2 + 1;
    else
        p.y = p.y*2;
    int2 tile_offset;
    tile_offset.x =       blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    gipuma_checkerboard_spatialPropClose_cu<T>(gs, p, tile_offset, iter);
}

template< typename T >
__global__ void gipuma_red_spatialPropFar_cu(GlobalState &gs, int iter)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2 + 1;
    else
        p.y = p.y*2;
    int2 tile_offset;
    tile_offset.x =       blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    gipuma_checkerboard_spatialPropFar_cu<T>(gs, p, tile_offset, iter);
}

template< typename T >
__global__ void gipuma_red_planeRefine_cu(GlobalState &gs, int iter)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    if (threadIdx.x%2==0)
        p.y = p.y*2 + 1;
    else
        p.y = p.y*2;
    int2 tile_offset;
    tile_offset.x =       blockIdx.x * blockDim.x - WIN_RADIUS_W;
    tile_offset.y = 2.0 * blockIdx.y * blockDim.y - WIN_RADIUS_H;
    gipuma_checkerboard_planeRefinement_cu<T>(gs, p, tile_offset, iter);
}

template< typename T >
void gipuma(GlobalState &gs)
{
#ifdef SHARED
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#else
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif

    int rows = gs.cameras->rows;
    int cols = gs.cameras->cols;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc ( &gs.cs, rows*cols*sizeof( curandState ) ));

    //int SHARED_SIZE_W_host;
#ifndef SHARED_HARDCODED
    int blocksize_w = gs.params->box_hsize + 1; // +1 for the gradient computation
    int blocksize_h = gs.params->box_vsize + 1; // +1 for the gradient computation
    WIN_RADIUS_W = (blocksize_w) / (2);
    WIN_RADIUS_H = (blocksize_h) / (2);

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W/2);
    TILE_W = BLOCK_W;
    TILE_H = BLOCK_H * 2;
    SHARED_SIZE_W_m  = (TILE_W + WIN_RADIUS_W * 2);
    SHARED_SIZE_H = (TILE_H + WIN_RADIUS_H * 2);
    SHARED_SIZE = (SHARED_SIZE_W_m * SHARED_SIZE_H);
    cudaMemcpyToSymbol (SHARED_SIZE_W, &SHARED_SIZE_W_m, sizeof(SHARED_SIZE_W_m));
    //SHARED_SIZE_W_host = SHARED_SIZE_W_m;
#else
    //SHARED_SIZE_W_host = SHARED_SIZE;
#endif
    int shared_size_host = SHARED_SIZE;

    dim3 grid_size;
    grid_size.x=(cols+BLOCK_W-1)/BLOCK_W;
    grid_size.y=((rows/2)+BLOCK_H-1)/BLOCK_H;
    dim3 block_size;
    block_size.x=BLOCK_W;
    block_size.y=BLOCK_H;

    dim3 grid_size_initrand;
    grid_size_initrand.x=(cols+16-1)/16;
    grid_size_initrand.y=(rows+16-1)/16;
    dim3 block_size_initrand;
    block_size_initrand.x=16;
    block_size_initrand.y=16;

    //printf("Launching kernel with grid of size %d %d and block of size %d %d and shared size %d %d\nBlock %d %d and radius %d %d and tile %d %d\n",
           //grid_size.x,
           //grid_size.y,
           //block_size.x,
           //block_size.y,
           //SHARED_SIZE_W_host,
           //SHARED_SIZE_H,
           //BLOCK_W,
           //BLOCK_H,
           //WIN_RADIUS_W,
           //WIN_RADIUS_H,
           //TILE_W,
           //TILE_H
          //);
    //printf("Grid size initrand is grid: %d-%d block: %d-%d\n", grid_size_initrand.x, grid_size_initrand.y, block_size_initrand.x, block_size_initrand.y);

    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    int maxiter=gs.params->iterations;
    printf("Device memory used: %fMB\n", used/1000000.0f);
    printf("Blocksize is %dx%d\n", gs.params->box_hsize,gs.params->box_vsize);

    //int shared_memory_size = sizeof(float)  * SHARED_SIZE ;
    //printf("Computing depth\n");
    printf("Number of iterations is %d\n", maxiter);
    //gipuma_init_cu<T><<< (rows + BLOCK_H-1)/BLOCK_H, BLOCK_H>>>(gs);
    //gipuma_init_random<<< grid_size_initrand, block_size_initrand>>>(gs);
    gipuma_init_cu2<T><<< grid_size_initrand, block_size_initrand>>>(gs);
    //gipuma_initial_cost<T><<< grid_size_initrand, block_size_initrand>>>(gs);
    cudaEventRecord(start);
    //for (int it =0;it<gs.params.iterations; it++) {
    printf("Iteration ");
    for (int it =0;it<maxiter; it++) {
        printf("%d ", it+1);
#ifdef SMALLKERNEL
        //spatial propagation of 4 closest neighbors (1px up/down/left/right)
        gipuma_black_spatialPropClose_cu<T><<< grid_size, block_size, shared_size_host * sizeof(T)>>>(gs, it);
        cudaDeviceSynchronize();
    #ifdef EXTRAPOINTFAR
        //spatial propagation of 4 far away neighbors (5px up/down/left/right)
        gipuma_black_spatialPropFar_cu<T><<< grid_size, block_size, shared_size_host * sizeof(T)>>>(gs, it);
        cudaDeviceSynchronize();
    #endif
        //plane refinement
        gipuma_black_planeRefine_cu<T><<< grid_size, block_size, shared_size_host * sizeof(T)>>>(gs, it);
        cudaDeviceSynchronize();

        //spatial propagation of 4 closest neighbors (1px up/down/left/right)
        gipuma_red_spatialPropClose_cu<T><<< grid_size, block_size, shared_size_host * sizeof(T)>>>(gs, it);
        cudaDeviceSynchronize();
    #ifdef EXTRAPOINTFAR
        //spatial propagation of 4 far away neighbors (5px up/down/left/right)
        gipuma_red_spatialPropFar_cu<T><<< grid_size, block_size, shared_size_host * sizeof(T)>>>(gs, it);
        cudaDeviceSynchronize();
    #endif
        //plane refinement
        gipuma_red_planeRefine_cu<T><<< grid_size, block_size, shared_size_host * sizeof(T)>>>(gs, it);
        cudaDeviceSynchronize();
#else
        gipuma_black_cu<T><<< grid_size, block_size, shared_size_host * sizeof(T)>>>(gs, it);
        gipuma_red_cu<T><<< grid_size, block_size, shared_size_host * sizeof(T)>>>(gs, it);
#endif
    }
    printf("\n");
    //printf("Computing final disparity\n");
    gipuma_compute_disp<<< grid_size_initrand, block_size_initrand>>>(gs);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\t\tTotal time needed for computation: %f seconds\n", milliseconds/1000.f);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    // print results to file
    cudaFree (&gs.cs);
}

int runcuda(GlobalState &gs)
{
    //printf("Run cuda\n");
    if(gs.params->color_processing)
        gipuma<float4>(gs);
    else
        gipuma<float>(gs);
    return 0;
}
