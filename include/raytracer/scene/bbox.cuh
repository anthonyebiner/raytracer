#pragma once

#include "raytracer/util/misc.cuh"
#include "raytracer/scene/ray.cuh"
#include "Eigen/Dense"

using Eigen::Vector3f;
using std::min;
using std::max;

class BBox {
public:
  Vector3f maxp;
  Vector3f minp;
  Vector3f extent;

  __host__ BBox() {
    maxp = Vector3f(-INF_F, -INF_F, -INF_F);
    minp = Vector3f(INF_F, INF_F, INF_F);
    extent = maxp - minp;
  }

  __host__ explicit BBox(const Vector3f &p) : minp(p), maxp(p) {
    extent = maxp - minp;
  }

  __host__ BBox(const Vector3f &minp, const Vector3f &maxp) :
      minp(minp), maxp(maxp) {
    extent = maxp - minp;
  }

  __host__ void expand(const BBox &bbox) {
    minp.x() = min(minp.x(), bbox.minp.x());
    minp.y() = min(minp.y(), bbox.minp.y());
    minp.z() = min(minp.z(), bbox.minp.z());
    maxp.x() = max(maxp.x(), bbox.maxp.x());
    maxp.y() = max(maxp.y(), bbox.maxp.y());
    maxp.z() = max(maxp.z(), bbox.maxp.z());
    extent = maxp - minp;
  }

  __host__ void expand(const Vector3f &p) {
    minp.x() = min(minp.x(), p.x());
    minp.y() = min(minp.y(), p.y());
    minp.z() = min(minp.z(), p.z());
    maxp.x() = max(maxp.x(), p.x());
    maxp.y() = max(maxp.y(), p.y());
    maxp.z() = max(maxp.z(), p.z());
    extent = maxp - minp;
  }

  __host__ Vector3f centroid() const {
    return (minp + maxp) / 2.0f;
  }

  __host__ float surface_area() const {
    if (empty()) return 0.0;
    return 2 * (extent.x() * extent.z() +
                extent.x() * extent.y() +
                extent.y() * extent.z());
  }

  __host__ bool empty() const {
    return minp.x() > maxp.x() || minp.y() > maxp.y() || minp.z() > maxp.z();
  }

  __host__ bool intersect(Ray *ray) const {
    return intersect(minp, maxp, ray);
  }

  __device__ __host__ static bool intersect(Vector3f minp, Vector3f maxp, Ray *ray) {
    float tmin = 0.0, tmax = ray->hit_t;

    for (int d = 0; d < 3; ++d) {
      float t1 = (minp[d] - ray->origin[d]) * ray->inv_d[d];
      float t2 = (maxp[d] - ray->origin[d]) * ray->inv_d[d];

      tmin = min(max(t1, tmin), max(t2, tmin));
      tmax = max(min(t1, tmax), min(t2, tmax));
    }

    return tmin <= tmax && tmax > ray->min_t;
  }
};
