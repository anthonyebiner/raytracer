#pragma once

#include "Eigen/Dense"

using Eigen::Vector3f;

class SceneLight {
public:

  enum {
    AREA,
    POINT,
    DIRECTIONAL,
    INVALID,
  } type;

  union {
    struct {
      Vector3f radiance;
      Vector3f position;
      Vector3f direction;
      Vector3f dim_x;
      Vector3f dim_y;
      float area;
    } area;
    struct {
      Vector3f radiance;
      Vector3f position;
    } point;
    struct {
      Vector3f radiance;
      Vector3f dirToLight;
    } directional;
  };

  __host__ __device__ explicit SceneLight() : type(INVALID) {}

  __host__ __device__ SceneLight(SceneLight const &l) {
    type = l.type;
    switch (type) {
      case AREA: {
        area = l.area;
      }
      case POINT: {
        point = l.point;
      }
      case DIRECTIONAL: {
        directional = l.directional;
      }
      case INVALID:
        break;
    }
  }

  SceneLight &operator=(SceneLight const &l) {
    type = l.type;
    switch (type) {
      case AREA: {
        area = l.area;
      }
      case POINT: {
        point = l.point;
      }
      case DIRECTIONAL: {
        directional = l.directional;
      }
      case INVALID:
        break;
    }
    return *this;
  }

  __device__ Vector3f sample(const Vector3f &p, Vector3f *wi,
                             float *distToLight, curandState *rand_state) {
    switch (type) {
      case AREA: {
        Vector2f sample = CudaSampler2D::sample_grid(rand_state) - Vector2f(0.5f, 0.5f);
        Vector3f d = area.position + sample.x() * area.dim_x + sample.y() * area.dim_y - p;
        float cosTheta = d.dot(area.direction);
        float dist = d.norm();
        float sqDist = powf(dist, 2.f);
        *wi = d / dist;
        *distToLight = dist;
        float pdf = sqDist / (area.area * fabs(cosTheta));
        Vector3f color = area.radiance / pdf;
        return cosTheta < 0 ? color : Vector3f::Zero();
      }
      case POINT: {
        Vector3f d = point.position - p;
        *wi = d.normalized();
        *distToLight = d.norm();
        return point.radiance;
      }
      case DIRECTIONAL: {
        *wi = directional.dirToLight;
        *distToLight = INF_F;
        return directional.radiance;
      }
      case INVALID:
        break;
    }
  }

  __device__ bool is_delta_light() const {
    switch (type) {
      case AREA: {
        return false;
      }
      case POINT: {
        return true;
      }
      case DIRECTIONAL: {
        return true;
      }
      case INVALID:
        break;
    }
  }
};

class SceneLightFactor {
public:
  __host__ static SceneLight create_area(const Vector3f &radiance, const Vector3f &position, const Vector3f &direction,
                                         const Vector3f &dim_x, const Vector3f &dim_y) {
    auto light = SceneLight();
    light.type = SceneLight::AREA;
    light.area = {radiance, position, direction.normalized(), dim_x, dim_y, dim_x.norm() * dim_y.norm()};
    return light;
  }

  __host__ static SceneLight create_point(const Vector3f &radiance, const Vector3f &position) {
    auto light = SceneLight();
    light.type = SceneLight::POINT;
    light.point = {radiance, position};
    return light;
  }

  __host__ static SceneLight create_directional(const Vector3f &radiance, const Vector3f &direction) {
    auto light = SceneLight();
    light.type = SceneLight::DIRECTIONAL;
    light.directional = {radiance, (-direction).normalized()};
    return light;
  }
};