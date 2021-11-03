#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __linux__
#include <unistd.h>
const int cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
const int ncores = sysconf(_SC_NPROCESSORS_ONLN);
#else
const int cache_line_size = 64;
const int ncores = 8;
#endif

// Performance of this depends on if the compiler inlines the memcpy
// call (but hopefully any relevant compiler knows how to do that)
// #define WRITE_PIXEL(dest, color) memcpy((dest), (color), 3)

constexpr int img_width = 4000;
constexpr int img_height = 4000;
constexpr int max_iterations = 200;
constexpr char seq_file_dest[] = "seq-output.ppm";
constexpr char openmp_file_dest[] = "openmp-output.ppm";
constexpr char sse_file_dest[] = "sse-output.ppm";
constexpr char header_comment[] =  "# ";
constexpr int max_color_component_value = 255;

typedef unsigned char color[3];

color white = { 255, 255, 255 };

color black = { 0, 0, 0 };

// Intermediate representation used to check if a pixel should be
// drawn black or white
int pixel_matrix[img_width][img_height];

// Setting up coordinates
constexpr float min_re = -2.5;
constexpr float max_re = 1.5;
constexpr float min_im = -2.0;
constexpr float max_im = 2.0;

constexpr float pixel_width = (max_re - min_re) / img_width;
constexpr float pixel_height = (max_im - min_im) / img_height;

// The radius controls whether a pixel is inside or outside; using the
// squared value allows us to skip an unnecessary abs() call
constexpr float radius_sq = 4.0;
