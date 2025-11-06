#include "minmax.h"

// Perform Min-Max Normalization column-wise for a matrix

// void min_max_normalize_columns(float* matrix, float* normalized_matrix, size_t rows, size_t cols, float new_min, float new_max) {
//     for (size_t col = 0; col < cols; col++) {
//         float col_min = FLT_MAX;
//         float col_max = FLT_MIN;

//         // Find the min and max for the current column
//         for (size_t row = 0; row < rows; row++) {
//             float value = matrix[row * cols + col];
//             if (value < col_min) col_min = value;
//             if (value > col_max) col_max = value;
//         }

//         // Add to each row of the column the minumum
//         for (size_t row = 0; row < rows; row++) {
//             matrix[row * cols + col] -= col_min;
//         }

//         float col_min_abs = FLT_MAX;
//         float col_max_abs = FLT_MIN;

//         // Find the min and max for the current column
//         for (size_t row = 0; row < rows; row++) {
//             float value = matrix[row * cols + col];
//             if (value < col_min_abs) col_min_abs = value;
//             if (value > col_max_abs) col_max_abs = value;
//         }

//         for (size_t row = 0; row < rows; row++) {
//             float value = matrix[row * cols + col];
//             if (col_max != col_min) {
//                 normalized_matrix[row * cols + col] = new_min + ((value - col_min_abs) / (col_max_abs - col_min_abs)) * (new_max - new_min);
//             } else {
//                 normalized_matrix[row * cols + col] = new_min; // All values are the same
//             }
//         }
        
//     }
    
// }


float* min_max_normalize_columns(float* matrix, size_t rows, size_t cols, float new_min, float new_max) {
    // Allocate memory for the normalized matrix
    float* normalized_matrix = (float*)malloc(rows * cols * sizeof(float));
    if (normalized_matrix == NULL) {
        perror("Unable to allocate memory for normalized matrix");
        exit(1);
    }

    for (size_t col = 0; col < cols; col++) {
        float col_min = FLT_MAX;
        float col_max = FLT_MIN;

        // Find the min and max for the current column
        for (size_t row = 0; row < rows; row++) {
            float value = matrix[row * cols + col];
            if (value < col_min) col_min = value;
            if (value > col_max) col_max = value;
        }

        // Add to each row of the column the minimum
        if (col_max < 0.0000000001){
            for (size_t row = 0; row < rows; row++) {
                matrix[row * cols + col] -= col_min;
            }

            float col_min_abs = FLT_MAX;
            float col_max_abs = FLT_MIN;

            // Find the min and max for the current column
            for (size_t row = 0; row < rows; row++) {
                float value = matrix[row * cols + col];
                if (value < col_min_abs) col_min_abs = value;
                if (value > col_max_abs) col_max_abs = value;
            }

            // Normalize the values in the column
            for (size_t row = 0; row < rows; row++) {
                float value = matrix[row * cols + col];
                if (col_max != col_min) {
                    normalized_matrix[row * cols + col] = new_min + ((value - col_min_abs) / (col_max_abs - col_min_abs)) * (new_max - new_min);
                } else {
                    normalized_matrix[row * cols + col] = new_min; // All values are the same
                }
            }
        } else {
            // Normalize the values in the column
            for (size_t row = 0; row < rows; row++) {
                float value = matrix[row * cols + col];
                if (col_max != col_min) {
                    normalized_matrix[row * cols + col] = new_min + ((value - col_min) / (col_max - col_min)) * (new_max - new_min);
                } else {
                    normalized_matrix[row * cols + col] = new_min; // All values are the same
                }
            }
        }
    }

    return normalized_matrix;
}
