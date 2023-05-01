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

  explicit Primitive() : type(INVALID), bsdf(nullptr) {}

  Primitive(Primitive const &p) {
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

  RAYTRACER_HOST_DEVICE_FUNC bool intersect(Ray *ray, Intersection *isect) {
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

        if (t1 <= ray->min_t || t0 > ray->max_t) return false;

        if (t0 > ray->min_t) {
          ray->max_t = t0;
        } else {
          if (t1 > ray->max_t) return false;
          ray->max_t = t1;
        }

        isect->primitive = this;
        isect->ray = ray;
        isect->t = ray->max_t;
        isect->normal = ((ray->origin + ray->max_t * ray->direction) - sphere.origin).normalized();
        return true;
      }
      case TRIANGLE: {
        Vector3f e1 = triangle.p2 - triangle.p1;
        Vector3f e2 = triangle.p3 - triangle.p1;
        Vector3f s = ray->origin - triangle.p1;
        Vector3f s1 = ray->direction.cross(e2);
        Vector3f s2 = s.cross(e1);

        Vector3f c = Vector3f(s2.dot(e2), s1.dot(s), s2.dot(ray->direction)) / s1.dot(e1);
        float t = c.x();

        Vector3f b = {1 - c.y() - c.z(), c.y(), c.z()};

        if (t < 0 || t <= ray->min_t || t > ray->max_t || b.x() < 0 || b.y() < 0 || b.z() < 0) {
          return false;
        }
        ray->max_t = t;

        isect->primitive = this;
        isect->ray = ray;
        isect->t = ray->max_t;
        isect->normal = (b.x() * triangle.n1 + b.y() * triangle.n2 + b.z() * triangle.n3).normalized();
        return true;
      }
      case INVALID:
        return 0;
    }
  }

  BBox get_bbox() {
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
  static Primitive createSphere(const Vector3f &center, float radius, BSDF *bsdf) {
    auto primitive = Primitive();
    primitive.type = Primitive::SPHERE;
    primitive.bsdf = bsdf;
    primitive.sphere = {center, radius, radius * radius};
    return primitive;
  }

  static Primitive
  createTriangle(const Vector3f &p1, const Vector3f &p2, const Vector3f &p3, const Vector3f &n1, const Vector3f &n2,
                 const Vector3f &n3, BSDF *bsdf) {
    auto primitive = Primitive();
    primitive.type = Primitive::TRIANGLE;
    primitive.bsdf = bsdf;
    primitive.triangle = {p1, p2, p3, n1, n2, n3};
    return primitive;
  }
};
