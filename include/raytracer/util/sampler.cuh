#pragma once

#include "curand_kernel.h"
#include "raytracer/util/misc.cuh"
#include "Eigen/Dense"

using Eigen::Vector2f;
using Eigen::Vector3f;

class CudaSampler1D {
public:
  __device__ static bool coin_flip(curandState *rand_state, float p) {
    return curand_uniform(rand_state) < p;
  }
};

class CudaSampler2D {
public:
  __device__ static Vector2f sample_grid(curandState *rand_state) {
    return {curand_uniform(rand_state), curand_uniform(rand_state)};
  }
};

class CudaSampler3D {
public:
  __device__ static Vector3f sample_grid(curandState *rand_state) {
    return {curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state)};
  }

  __device__ static Vector3f sample_sphere(curandState *rand_state) {
    float z = curand_uniform(rand_state) * 2 - 1;
    float sinTheta = sqrtf(max(0.0f, 1.0f - z * z));

    float phi = 2.0f * PI * curand_uniform(rand_state);

    return {cosf(phi) * sinTheta, sinf(phi) * sinTheta, z};
  }

  __device__ static Vector3f sample_hemisphere(curandState *rand_state) {
    float Xi1 = curand_uniform(rand_state);
    float Xi2 = curand_uniform(rand_state);

    float theta = acosf(Xi1);
    float phi = 2.0f * PI * Xi2;

    float xs = sinf(theta) * cosf(phi);
    float ys = sinf(theta) * sinf(phi);
    float zs = cosf(theta);

    return {xs, ys, zs};
  }

  __device__ static Vector3f sample_cosine_weighted_hemisphere(curandState *rand_state) {
    float Xi1 = curand_uniform(rand_state);
    float Xi2 = curand_uniform(rand_state);

    float r = sqrtf(Xi1);
    float theta = 2.0f * PI * Xi2;
    return {r * cosf(theta), r * sinf(theta), sqrtf(1 - Xi1)};
  }
};
