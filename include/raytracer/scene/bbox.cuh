#pragma once

#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/scene/ray.cuh"
#include "raytracer/util/misc.cuh"

class BBox {
public:
  Vector3f maxp;
  Vector3f minp;
  Vector3f extent;

  BBox() {
    maxp = {-INF_F, -INF_F, -INF_F};
    minp = {INF_F, INF_F, INF_F};
    extent = maxp - minp;
  }

  explicit BBox(const Vector3f &p) : minp(p), maxp(p) {
    extent = maxp - minp;
  }

  BBox(const Vector3f &minp, const Vector3f &maxp) :
      minp(minp), maxp(maxp) {
    extent = maxp - minp;
  }

  void expand(const BBox &bbox) {
    minp.x = fminf(minp.x, bbox.minp.x);
    minp.y = fminf(minp.y, bbox.minp.y);
    minp.z = fminf(minp.z, bbox.minp.z);
    maxp.x = fmaxf(maxp.x, bbox.maxp.x);
    maxp.y = fmaxf(maxp.y, bbox.maxp.y);
    maxp.z = fmaxf(maxp.z, bbox.maxp.z);
    extent = maxp - minp;
  }

  void expand(const Vector3f &p) {
    minp.x = fminf(minp.x, p.x);
    minp.y = fminf(minp.y, p.y);
    minp.z = fminf(minp.z, p.z);
    maxp.x = fmaxf(maxp.x, p.x);
    maxp.y = fmaxf(maxp.y, p.y);
    maxp.z = fmaxf(maxp.z, p.z);
    extent = maxp - minp;
  }

  Vector3f centroid() const {
    return (minp + maxp) / 2.0f;
  }

  float surface_area() const {
    if (empty()) return 0.0;
    return 2 * (extent.x * extent.z +
                extent.x * extent.y +
                extent.y * extent.z);
  }

  bool empty() const {
    return minp.x > maxp.x || minp.y > maxp.y || minp.z > maxp.z;
  }

  bool intersect(const Ray &ray) const {
    return intersect(minp, maxp, ray);
  }

  RAYTRACER_DEVICE_FUNC static bool intersect(Vector3f minp, Vector3f maxp, const Ray &ray) {
    float tmin = 0.0, tmax = ray.max_t;

    for (int d = 0; d < 3; ++d) {
      float t1 = (minp[d] - ray.origin[d]) * ray.inv_d[d];
      float t2 = (maxp[d] - ray.origin[d]) * ray.inv_d[d];

      tmin = fminf(fmaxf(t1, tmin), fmaxf(t2, tmin));
      tmax = fmaxf(fminf(t1, tmax), fminf(t2, tmax));
    }

    return tmin <= tmax && tmax > ray.min_t;
  }
};
