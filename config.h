#pragma once
#define MAX_IMAGES 512

/*#define SHARED_HARDCODED*/
#ifdef SHARED_HARDCODED
#define BLOCKSIZE_W 15 + 1 // +1 for the gradient computation
#define BLOCKSIZE_H 15 + 1 // +1 for the gradient computation
#define WIN_RADIUS_W (BLOCKSIZE_W) / (2)
#define WIN_RADIUS_H (BLOCKSIZE_H) / (2)

#define BLOCK_W 32
#define BLOCK_H (BLOCK_W/2)
#define TILE_W BLOCK_W
#define TILE_H BLOCK_H * 2
#define TILE_SIZE TILE_W * TILE_H
#define SHARED_SIZE_W (TILE_W + WIN_RADIUS_W * 2)
#define SHARED_SIZE_H (TILE_H + WIN_RADIUS_H * 2)
#define SHARED_SIZE (SHARED_SIZE_W * SHARED_SIZE_H)
#endif

#define REFERENCE 0
#define MAXCOST 1000.0f

#define FORCEINLINE __forceinline__
//#define FORCEINLINE

#define pow2(x) ((x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define get_pow2_norm(x,y) (pow2(x)+pow2(y))
#define get_normf(x,y) (sqrtf((pow2(x)+pow2(y))))
#define get_normf3(v) (sqrtf((pow2(v.x)+pow2(v.y) + pow2(v.z))))
#define get_pow2_normf3(v) ((pow2(v.x)+pow2(v.y) + pow2(v.z)))

// Vector operations
#define dot4(v0,v1) v0.x * v1.x + \
                    v0.y * v1.y + \
                    v0.z * v1.z
#define mul4(v,k) \
v->x = v->x * k; \
v->y = v->y * k; \
v->z = v->z * k;

#define vecdiv4(v,k) \
v->x = v->x / k; \
v->y = v->y / k; \
v->z = v->z / k;

#define vecdiv42(v,k) \
v.x = v.x / k; \
v.y = v.y / k; \
v.z = v.z / k;


#define vecdiv(v,k) \
v[0] = v[0] / k; \
v[1] = v[1] / k; \
v[2] = v[2] / k;

#define get_pow2_normv3(x,y) (pow2(x)+pow2(y))

#define negate(v) mul(v,-1.0f)
#define negate4(v) v->x = -v->x; \
                   v->y = -v->y; \
                   v->z = -v->z;

#define sub(v0,v1) v0.x = v0.x - v1.x; \
                   v0.y = v0.y - v1.y; \
                   v0.z = v0.z - v1.z;
#define subout(v0,v1,vout) vout.x = v0.x - v1.x; \
                   vout.y = v0.y - v1.y; \
                   vout.z = v0.z - v1.z;

#define outer_product(v0,v1, out) \
out[0] = v0[0] * v1[0]; \
out[1] = v0[0] * v1[1]; \
out[2] = v0[0] * v1[2]; \
out[3] = v0[1] * v1[0]; \
out[4] = v0[1] * v1[1]; \
out[5] = v0[1] * v1[2]; \
out[6] = v0[2] * v1[0]; \
out[7] = v0[2] * v1[1]; \
out[8] = v0[2] * v1[2];

#define outer_product4(v0,v1, out) \
out[0] = v0.x * v1.x; \
out[1] = v0.x * v1.y; \
out[2] = v0.x * v1.z; \
out[3] = v0.y * v1.x; \
out[4] = v0.y * v1.y; \
out[5] = v0.y * v1.z; \
out[6] = v0.z * v1.x; \
out[7] = v0.z * v1.y; \
out[8] = v0.z * v1.z;
// Matrix operations
#define matadd(m,k)   m[0] = m[0] + k; \
                   m[1] = m[1] + k; \
                   m[2] = m[2] + k; \
                   m[3] = m[3] + k; \
                   m[4] = m[4] + k; \
                   m[5] = m[5] + k; \
                   m[6] = m[6] + k; \
                   m[7] = m[7] + k; \
                   m[8] = m[8] + k;

#define matsub(m,k)   m[0] = m[0] - k; \
                   m[1] = m[1] - k; \
                   m[2] = m[2] - k; \
                   m[3] = m[3] - k; \
                   m[4] = m[4] - k; \
                   m[5] = m[5] - k; \
                   m[6] = m[6] - k; \
                   m[7] = m[7] - k; \
                   m[8] = m[8] - k;

#define matmatsub(m0, m1) \
m0[0] = m0[0] - m1[0]; \
m0[1] = m0[1] - m1[1]; \
m0[2] = m0[2] - m1[2]; \
m0[3] = m0[3] - m1[3]; \
m0[4] = m0[4] - m1[4]; \
m0[5] = m0[5] - m1[5]; \
m0[6] = m0[6] - m1[6]; \
m0[7] = m0[7] - m1[7]; \
m0[8] = m0[8] - m1[8];

#define matmatsub2(m0, m1) \
m1[0] = m0[0] - m1[0]; \
m1[1] = m0[1] - m1[1]; \
m1[2] = m0[2] - m1[2]; \
m1[3] = m0[3] - m1[3]; \
m1[4] = m0[4] - m1[4]; \
m1[5] = m0[5] - m1[5]; \
m1[6] = m0[6] - m1[6]; \
m1[7] = m0[7] - m1[7]; \
m1[8] = m0[8] - m1[8];


#define matdivide(m,k) \
m[0] = m[0] / k; \
m[1] = m[1] / k; \
m[2] = m[2] / k; \
m[3] = m[3] / k; \
m[4] = m[4] / k; \
m[5] = m[5] / k; \
m[6] = m[6] / k; \
m[7] = m[7] / k; \
m[8] = m[8] / k;

#define matvecmul4noz(m, v, out) \
out->x = \
m [0] * v.x + \
m [1] * v.y + \
m [2];\
out->y = \
m [3] * v.x + \
m [4] * v.y + \
m [5]; \
out->z = \
m [6] * v.x + \
m [7] * v.y + \
m [8];

#define matvecmul4(m, v, out) \
out->x = \
m [0] * v.x + \
m [1] * v.y + \
m [2] * v.z; \
out->y = \
m [3] * v.x + \
m [4] * v.y + \
m [5] * v.z; \
out->z = \
m [6] * v.x + \
m [7] * v.y + \
m [8] * v.z;
#define matvecmul4div(m, v, out,d) \
out->x = \
(m [0] * v.x + \
m [1] * v.y + \
m [2] * v.z)/d; \
out->y = \
(m [3] * v.x + \
m [4] * v.y + \
m [5] * v.z)/d; \
out->z = \
(m [6] * v.x + \
m [7] * v.y + \
m [8] * v.z)/d;

#define matvecmul(m, v, out) \
out[0] = \
m [0] * v[0] + \
m [1] * v[1] + \
m [2] * v[2]; \
out[1] = \
m [3] * v[0] + \
m [4] * v[1] + \
m [5] * v[2]; \
out[2] = \
m [6] * v[0] + \
m [7] * v[1] + \
m [8] * v[2];

#define matmul_cu(m0, m1, out) \
out[0] = \
m0 [0] * m1[0] + \
m0 [1] * m1[0+3] + \
m0 [2] * m1[0+6]; \
out[1] = \
m0 [0] * m1[1] + \
m0 [1] * m1[1+3] + \
m0 [2] * m1[1+6]; \
out[2] = \
m0 [0] * m1[2] + \
m0 [1] * m1[2+3] + \
m0 [2] * m1[2+6]; \
out[3] = \
m0 [3] * m1[0] + \
m0 [4] * m1[0+3] + \
m0 [5] * m1[0+6]; \
out[4] = \
m0 [3] * m1[1] + \
m0 [4] * m1[1+3] + \
m0 [5] * m1[1+6]; \
out[5] = \
m0 [3] * m1[2] + \
m0 [4] * m1[2+3] + \
m0 [5] * m1[2+6]; \
out[6] = \
m0 [6] * m1[0] + \
m0 [7] * m1[0+3] + \
m0 [8] * m1[0+6]; \
out[7] = \
m0 [6] * m1[1] + \
m0 [7] * m1[1+3] + \
m0 [8] * m1[1+6]; \
out[8] = \
m0 [6] * m1[2] + \
m0 [7] * m1[2+3] + \
m0 [8] * m1[2+6];

// texture utils

#define texat(tex,x, y) \
 tex2D<float>(tex, x+0.5f, y+0.5f)
#define texatf4(tex,x, y) \
 tex2D<float4>(tex, x+0.5f, y+0.5f)

#define texatpt(tex,pt) \
texat(tex, pt[0], pt[1])
#define texatpt4(tex,pt) \
texat(tex, pt.x, pt.y)
#define texatptf4(tex,pt) \
texatf4(tex, pt.x, pt.y)

#define colorDifferenceL1_macro(x,y) \
fabsf(x-y)

#define print_vector(v,string) \
printf("%s is \tnp.array([%e, %e ,%e])\n", string, v[0],v[1],v[2]);

#define print_vector4(v,string) \
printf("%s is \tnp.array([%f, %f ,%f])\n", string, v.x,v.y,v.z);

#define print_matrix(m,string) \
printf("%s is  [[%.4e, %.4e, %.4e], \n\t [%.4e, %.4e, %.4e], \n\t [%.4e, %.4e, %.4e]]\n", string, m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8]);
