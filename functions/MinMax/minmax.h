#ifndef MINMAX_H
#define MINMAX_H

#include <stddef.h> // For size_t
#include <float.h>  // For FLT_MAX and FLT_MIN
#include <stdlib.h>
#include <stdio.h>


// Function prototypes
float* min_max_normalize_columns(float* matrix, size_t rows, size_t cols, float new_min, float new_max);

#endif // MINMAX_H