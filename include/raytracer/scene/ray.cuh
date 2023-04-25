#pragma once

#include <utility>
#include "raytracer/util/misc.cuh"
#include "raytracer/scene/bsdf.cuh"
#include "Eigen/Dense"

using Eigen::Vector3f;

struct Primitive;

struct Ray {
  Vector3f origin;
  Vector3f direction;
  Vector3f inv_d;

  float min_t;
  float hit_t;

  Vector3f hit_n;
  Primitive *hit_p;

  __device__ __host__ Ray(Vector3f origin, const Vector3f &direction)
      : origin(std::move(origin)), direction(direction.normalized()), min_t(EPS_F), hit_t(INF_F),
        inv_d(direction.cwiseInverse()), hit_p(nullptr), hit_n() {}

  __device__ __host__ Ray(Vector3f origin, const Vector3f &direction, float max_t)
      : origin(std::move(origin)), direction(direction.normalized()), min_t(EPS_F), hit_t(max_t),
        inv_d(direction.cwiseInverse()), hit_p(nullptr), hit_n() {}
};
