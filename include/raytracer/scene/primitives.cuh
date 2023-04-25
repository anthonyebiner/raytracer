#pragma once

#include <utility>

#include "raytracer/scene/bbox.cuh"
#include "raytracer/scene/ray.cuh"
#include "raytracer/scene/bsdf.cuh"
#include "Eigen/Dense"

using Eigen::Vector3f;

struct Primitive {
  enum {
    INVALID,
    SPHERE,
    TRIANGLE,
  } type;
  BSDF *bsdf;
  union {
    struct {
      Vector3f origin;
      float r;
      float r2;
    } sphere;
    struct {
      Vector3f p1, p2, p3;
      Vector3f n1, n2, n3;
    } triangle;
  };

  __host__ explicit Primitive() : type(INVALID), bsdf(nullptr) {}

  __host__ Primitive(Primitive const &p) {
    type = p.type;
    bsdf = p.bsdf;
    switch (type) {
      case SPHERE: {
        sphere = p.sphere;
      }
      case TRIANGLE: {
        triangle = p.triangle;
      }
      case INVALID:
        break;
    }
  }

  Primitive &operator=(Primitive const &p) {
    type = p.type;
    bsdf = p.bsdf;
    switch (type) {
      case SPHERE: {
        sphere = p.sphere;
      }
      case TRIANGLE: {
        triangle = p.triangle;
      }
      case INVALID:
        break;
    }
    return *this;
  }

  __device__ __host__ bool intersect(Ray *ray) {
    switch (type) {
      case SPHERE: {
        Vector3f a = sphere.origin - ray->origin;
        float b = a.dot(ray->direction);
        if (b < 0) return false;
        float c = a.dot(a) - b * b;

        if (c > sphere.r2) return false;
        float disc = sqrtf(sphere.r2 - c);

        float t0 = b - disc;
        float t1 = b + disc;

        if (t1 < ray->min_t || t0 > ray->hit_t) return false;

        if (t0 > ray->min_t) {
          ray->hit_t = t0;
        } else {
          if (t1 > ray->hit_t) return false;
          ray->hit_t = t1;
        }

        auto point = ray->origin + ray->hit_t * ray->direction;
        ray->hit_n = (point - sphere.origin).normalized();
        ray->hit_p = this;
        return true;
      }
      case TRIANGLE: {
        auto e1 = triangle.p2 - triangle.p1;
        auto e2 = triangle.p3 - triangle.p1;
        auto s = ray->origin - triangle.p1;
        auto s1 = ray->direction.cross(e2);
        auto s2 = s.cross(e1);

        auto c = Vector3f(s2.dot(e2), s1.dot(s), s2.dot(ray->direction)) / s1.dot(e1);
        auto t = c.x();

        auto b = Vector3f(1 - c.y() - c.z(), c.y(), c.z());

        if (t < 0 || t < ray->min_t || t > ray->hit_t || b.x() < 0 || b.y() < 0 || b.z() < 0) {
          return false;
        }
        ray->hit_t = t;
        ray->hit_n = b.x() * triangle.n1 + b.y() * triangle.n2 + b.z() * triangle.n3;
        ray->hit_p = this;
        return true;
      }
      case INVALID:
        return 0;
    }
  }

  __host__ BBox get_bbox() {
    switch (type) {
      case SPHERE: {
        return {sphere.origin - Vector3f(sphere.r, sphere.r, sphere.r),
                sphere.origin + Vector3f(sphere.r, sphere.r, sphere.r)};
      }
      case TRIANGLE: {
        BBox bbox = BBox(triangle.p1);
        bbox.expand(triangle.p2);
        bbox.expand(triangle.p3);
        return bbox;
      }
      case INVALID:
        return {};
    }
  }
};

class PrimitiveFactory {
public:
  __host__ static Primitive createSphere(const Vector3f &center, float radius, BSDF *bsdf) {
    auto primitive = Primitive();
    primitive.type = Primitive::SPHERE;
    primitive.bsdf = bsdf;
    primitive.sphere = {center, radius, radius * radius};
    return primitive;
  }

  __host__ static Primitive
  createTriangle(const Vector3f &p1, const Vector3f &p2, const Vector3f &p3, const Vector3f &n1, const Vector3f &n2,
                 const Vector3f &n3, BSDF *bsdf) {
    auto primitive = Primitive();
    primitive.type = Primitive::TRIANGLE;
    primitive.bsdf = bsdf;
    primitive.triangle = {p1, p2, p3, n1, n2, n3};
    return primitive;
  }
};
