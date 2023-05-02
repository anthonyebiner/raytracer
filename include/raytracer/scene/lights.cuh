#pragma once

#include "raytracer/util/misc.cuh"
#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/util/sampler.cuh"
#include "raytracer/linalg/Vector2f.cuh"

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
      Vector3f dir_to_light;
    } directional;
  };

  RAYTRACER_DEVICE_FUNC explicit SceneLight() : type(INVALID) {}

  RAYTRACER_DEVICE_FUNC SceneLight(const SceneLight &l) {
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

  SceneLight &operator=(const SceneLight &l) {
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

  RAYTRACER_DEVICE_FUNC Vector3f sample(const Vector3f &hit_point, Vector3f *o_in,
                                        float *dist_to_light, float *pdf, uint *seed) {
    switch (type) {
      case AREA: {
        Vector2f sample = Sampler2D::sample_grid(seed) - Vector2f(0.5f, 0.5f);
        Vector3f d = area.position + sample.x * area.dim_x + sample.y * area.dim_y - hit_point;
        float dist = d.norm();
        float dist2 = pow(dist, 2);
        float cos_theta = d.dot(area.direction) / dist;
        *o_in = d / dist;
        *dist_to_light = dist;
        *pdf = dist2 / (area.area * fabs(cos_theta));
        return cos_theta < 0 ? area.radiance : Vector3f(0, 0, 0);
      }
      case POINT: {
        Vector3f d = point.position - hit_point;
        *o_in = d.unit();
        *dist_to_light = d.norm();
        *pdf = 1;
        return point.radiance;
      }
      case DIRECTIONAL: {
        *o_in = directional.dir_to_light;
        *dist_to_light = INF_F;
        *pdf = 1;
        return directional.radiance;
      }
      case INVALID:
        break;
    }
  }

  RAYTRACER_DEVICE_FUNC bool is_delta_light() const {
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

class SceneLightFactory {
public:
  static SceneLight create_area(const Vector3f &radiance, const Vector3f &position, const Vector3f &direction,
                                const Vector3f &dim_x, const Vector3f &dim_y) {
    auto light = SceneLight();
    light.type = SceneLight::AREA;
    light.area = {radiance, position, direction.unit(), dim_x, dim_y, dim_x.norm() * dim_y.norm()};
    return light;
  }

  static SceneLight create_point(const Vector3f &radiance, const Vector3f &position) {
    auto light = SceneLight();
    light.type = SceneLight::POINT;
    light.point = {radiance, position};
    return light;
  }

  static SceneLight create_directional(const Vector3f &radiance, const Vector3f &direction) {
    auto light = SceneLight();
    light.type = SceneLight::DIRECTIONAL;
    light.directional = {radiance, (-direction).unit()};
    return light;
  }
};