#ifndef __MIA_UTILS__
#define __MIA_UTILS__

#include <stdlib.h>
#include <stdio.h>
#include <glib-2.0/glib.h>

void alloc_vector
(float **vector,   /* vector */
 long  n);          /* size */

void alloc_matrix
(float ***matrix,  /* matrix */
 long  nx,         /* size in x direction */
 long  ny);         /* size in y direction */

void alloc_matrix_d
(double ***matrix,  /* matrix */
 long  nx,         /* size in x direction */
 long  ny);         /* size in y direction */

/* allocates storage for matrix of size nx * ny */
void disalloc_vector

(float *vector,    /* vector */
 long  n);          /* size */

/* disallocates storage for a vector of size n */
void disalloc_vector

(float *vector,    /* vector */
 long  n);          /* size */


/*--------------------------------------------------------------------------*/

void disalloc_matrix

(float **matrix,   /* matrix */
 long  nx,         /* size in x direction */
 long  ny);         /* size in y direction */

void disalloc_matrix_d

(double **matrix,   /* matrix */
 long  nx,         /* size in x direction */
 long  ny);         /* size in y direction */
/* disallocates storage for matrix of size nx * ny */

#endif
