#pragma once


#if defined(__CUDACC__)
#define RAYTRACER_HOST_DEVICE_FUNC __host__ __device__
#define RAYTRACER_HOST_FUNC __host__
#else
#define RAYTRACER_HOST_DEVICE_FUNC
#define RAYTRACER_HOST_FUNC
#endif


#if defined(_WIN64) || defined(_WIN32)
#define uint unsigned int
#endif


#define PI (3.14159265358979323f)
#define EPS_F (0.00001f)
#define INF_F INFINITY
#define MAX_BVH_STACK 32

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


template<typename T>
RAYTRACER_HOST_DEVICE_FUNC inline T radians(T deg) {
  return deg * (PI / 180);
}

template<typename T>
RAYTRACER_HOST_DEVICE_FUNC inline T degrees(T rad) {
  return rad * (180 / PI);
}

template<typename T>
RAYTRACER_HOST_DEVICE_FUNC inline T clamp(T x, T lo, T hi) {
  return std::min(std::max(x, lo), hi);
}

void print_progress(double percentage) {
  int val = (int) (percentage * 100);
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}
