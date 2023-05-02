#pragma once

#include <utility>
#include "raytracer/util/misc.cuh"
#include "raytracer/scene/bsdf.cuh"
#include "Eigen/Dense"

using Eigen::Vector3f;
using Eigen::Matrix3f;


struct Ray {
  Vector3f origin;
  Vector3f direction;
  Vector3f inv_d;

  float min_t;
  float max_t;

  RAYTRACER_DEVICE_FUNC Ray() {};

  RAYTRACER_DEVICE_FUNC Ray(const Vector3f &origin, const Vector3f &direction)
      : origin(origin), direction(direction.normalized()), min_t(EPS_F), max_t(INF_F),
        inv_d(direction.cwiseInverse()) {}

  RAYTRACER_DEVICE_FUNC Ray(const Vector3f &origin, const Vector3f &direction, float max_t)
      : origin(origin), direction(direction.normalized()), min_t(EPS_F), max_t(max_t),
        inv_d(direction.cwiseInverse()) {}
};


struct Primitive;

struct Intersection {
  Primitive *primitive;
  Ray *ray;
  float t;
  Vector3f normal;

  Matrix3f o2w;
  Matrix3f w2o;

  Vector3f hit_point;
  Vector3f o_out;

  RAYTRACER_DEVICE_FUNC void compute() {
    normal = normal.normalized();
    make_coord_space(o2w, normal);
    w2o = o2w.transpose();

    hit_point = ray->origin + ray->direction * t;
    o_out = (w2o * (-ray->direction)).normalized();
  }
};
