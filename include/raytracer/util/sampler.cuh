#pragma once

#include "curand_kernel.h"
#include "raytracer/util/misc.cuh"
#include <random>
#include "Eigen/Dense"

using Eigen::Vector2f;
using Eigen::Vector3f;

static inline __host__ __device__ float random_float(uint *state) {
  uint x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return float(x) / float(std::numeric_limits<uint>::max());
}

class Sampler1D {
public:
  __device__ __host__ static float random(uint *seed) {
    return random_float(seed);
  }

  __device__ __host__ static bool coin_flip(uint *seed, float p) {
    return random_float(seed) < p;
  }
};

class Sampler2D {
public:
  __device__ __host__ static Vector2f sample_grid(uint *seed) {
    return {random_float(seed), random_float(seed)};
  }
};

class Sampler3D {
public:
  __device__ __host__ static Vector3f sample_grid(uint *seed) {
    return {random_float(seed), random_float(seed), random_float(seed)};
  }

  __device__ __host__ static Vector3f sample_sphere(uint *seed) {
    float z = random_float(seed) * 2 - 1;
    float sinTheta = sqrtf(max(0.0f, 1.0f - z * z));

    float phi = 2.0f * PI * random_float(seed);

    return {cosf(phi) * sinTheta, sinf(phi) * sinTheta, z};
  }

  __device__ __host__ static Vector3f sample_hemisphere(uint *seed) {
    float Xi1 = random_float(seed);
    float Xi2 = random_float(seed);

    float theta = acosf(Xi1);
    float phi = 2.0f * PI * Xi2;

    float xs = sinf(theta) * cosf(phi);
    float ys = sinf(theta) * sinf(phi);
    float zs = cosf(theta);

    return {xs, ys, zs};
  }

  __device__ __host__ static Vector3f sample_cosine_weighted_hemisphere(uint *seed) {
    float Xi1 = random_float(seed);
    float Xi2 = random_float(seed);

    float a = acosf(sqrtf(Xi1));
    float b = 2.0f * PI * Xi2;
    return {sinf(a) * cosf(b), sinf(a) * sinf(b), cosf(a)};
  }
};
