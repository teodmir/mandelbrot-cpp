#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "utils.hpp"

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <ctime>

void write_ppm(const char* filename)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("error: unable to open file %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }
    /* Write header to file */
    fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",
            header_comment,
            img_width,
            img_height,
            max_color_component_value);
    // Write to file based on pixel data
    for (size_t y = 0; y < img_height; y++) {
        for (size_t x = 0; x < img_width; x++) {
            if (pixel_matrix[x][y])
                fwrite(black, 1, 3, fp);
            else
                fwrite(white, 1, 3, fp);
        }
    }
    printf("Wrote to output file: %s\n", filename);
    fclose(fp);
}

void run_openmp()
{
#pragma omp parallel for schedule(dynamic)
    for (size_t x = 0; x < img_height; x++) {
        for (size_t y = 0; y < img_width; y++) {
            float c_im = min_im + y * pixel_height;
            float c_re = min_re + x * pixel_width;
            float z_re = c_re;
            float z_im = c_im;
            bool isInside = true;
            int n;
            for (n = 0; n < max_iterations; n++) {
                float z_re_sq = z_re * z_re;
                float z_im_sq = z_im * z_im;
                if(z_re_sq + z_im_sq > radius_sq) {
                    isInside = false;
                    break;
                }
                z_im = 2 * z_re * z_im + c_im;
                z_re = z_re_sq - z_im_sq + c_re;
            }
            // pixel_matrix[x][y] = isInside;
            pixel_matrix[x][y] = isInside;
        }
    }
}

void run_seq()
{
    for (size_t y = 0; y < img_height; y++) {
        float c_im = min_im + y * pixel_height;
        for (size_t x = 0; x < img_width; x++) {
            float c_re = min_re + x * pixel_width;
            float z_re = c_re;
            float z_im = c_im;
            bool isInside = true;
            int n;
            for (n = 0; n < max_iterations; n++) {
                float z_re_sq = z_re * z_re;
                float z_im_sq = z_im * z_im;
                if(z_re_sq + z_im_sq > radius_sq) {
                    isInside = false;
                    break;
                }
                z_im = 2 * z_re * z_im + c_im;
                z_re = z_re_sq - z_im_sq + c_re;
            }
            pixel_matrix[x][y] = isInside;
        }
    }
}

__m128 zero_v = _mm_setzero_ps();
#define SSE_ALL_TRUE(v) _mm_movemask_ps(_mm_cmpeq_ps((v), zero_v)) == 0xF

// We make use of overloaded arithmetic operators for intrinsics
// provided by the compiler here
void run_sse()
{
    if (img_width % 4 != 0 || img_height % 4 != 0) {
        printf("Image dimensions required to be divisible by 4\n");
        exit(EXIT_FAILURE);
    }
    __m128 min_re_v = _mm_set1_ps(min_re);
    __m128 min_im_v = _mm_set1_ps(min_im);
    __m128 pixel_width_v = _mm_set1_ps(pixel_width);
    __m128 pixel_height_v = _mm_set1_ps(pixel_height);
    __m128 radius_sq_v = _mm_set1_ps(radius_sq);
    __m128 two_v = _mm_set1_ps(2);

    for (int y = 0; y < img_height; y += 4) {
        __m128 y_v = _mm_set_ps(y + 3, y + 2, y + 1, y);
		__m128 c_im_v = _mm_add_ps(_mm_mul_ps(y_v , pixel_height_v) , min_im_v);
        for (int x = 0; x < img_width; x++) {
            __m128 x_v = _mm_set1_ps(x);
			__m128 c_re_v = _mm_add_ps(_mm_mul_ps(x_v , pixel_width_v) , min_re_v);
            __m128 z_re_v = c_re_v;
            __m128 z_im_v = c_im_v;
            __m128 isInside_v = _mm_set1_ps(1.0);
            for (int n = 0; n < max_iterations; n++) {
                __m128 z_re_sq_v = _mm_mul_ps(z_re_v, z_re_v);
                __m128 z_im_sq_v = _mm_mul_ps(z_im_v , z_im_v);
                // Can't break early here, as with the other solutions
                isInside_v = _mm_cmple_ps(_mm_add_ps(z_re_sq_v , z_im_sq_v), radius_sq_v);
                if (SSE_ALL_TRUE(isInside_v)) {
                    // Avoiding the _mm_store_ps call and temporary
                    // array assignment
                    pixel_matrix[x][y] = false;
                    pixel_matrix[x][y + 1] = false;
                    pixel_matrix[x][y + 2] = false;
                    pixel_matrix[x][y + 3] = false;
                    goto END_ITER;
                }
				z_im_v = _mm_add_ps(_mm_mul_ps(two_v , _mm_mul_ps(z_re_v , z_im_v)), c_im_v);
                z_re_v = _mm_add_ps(c_re_v, _mm_sub_ps(z_re_sq_v , z_im_sq_v));
            }
            // The "boolean" results in isInside_v are still of type
            // float, so a cast is necessary here by using a temporary
            // float array
            float tmp[4];
            _mm_store_ps(tmp, isInside_v);
            memcpy(&pixel_matrix[x][y], &tmp, sizeof(tmp));
        END_ITER: ;
        }
    }
}

int main(int argc, char *argv[])
{
    clock_t start, end;
    double start_omp, end_omp;

    start = clock();
    run_seq();
    end = clock();
    printf("seq: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    write_ppm(seq_file_dest);

    start = clock();
    run_sse();
    end = clock();
    printf("sse: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    write_ppm(sse_file_dest);

    start_omp = omp_get_wtime();
    run_openmp();
    end_omp = omp_get_wtime();
    printf("openmp: %f\n", (double)(end_omp - start_omp));
    write_ppm(openmp_file_dest);

    return EXIT_SUCCESS;
}
