#pragma once

#include "raytracer/scene/ray.cuh"
#include "Eigen/Dense"

using Eigen::Vector3f;
using Eigen::Matrix3f;


class Camera {
public:
  Vector3f origin;
  Matrix3f c2w;
  float vFov;
  float hFov;
  float nClip;
  float fClip;

  __host__ Camera(const Vector3f &look_from, const Vector3f &look_at, const Vector3f &up,
                  float vFov, float hFov, float nClip, float fClip) {
    this->nClip = nClip;
    this->fClip = fClip;
    this->vFov = vFov;
    this->hFov = hFov;
    this->origin = look_from;

    Vector3f dirToCamera = look_from - look_at;
    Vector3f screenXDir = up.cross(dirToCamera).normalized();
    Vector3f screenYDir = dirToCamera.cross(screenXDir).normalized();

    c2w.col(0) = screenXDir;
    c2w.col(1) = screenYDir;
    c2w.col(2) = dirToCamera.normalized();
  }

  __host__ __device__ Ray generate_ray(float x, float y) const {
    Vector3f camera_vector = Vector3f(
        (0.5f - x) * tanf(hFov * PI / 360) * 2,
        (0.5f - y) * tanf(vFov * PI / 360) * 2,
        -1);
    Ray world_ray = Ray(origin, (c2w * camera_vector).normalized());
    world_ray.min_t = nClip;
    world_ray.hit_t = fClip;
    return world_ray;
  }

  __host__ Camera *to_cuda() const {
    Camera *cam_d;

    cudaMalloc(&cam_d, sizeof(Camera));
    cudaMemcpy(cam_d, this, sizeof(Camera), cudaMemcpyHostToDevice);

    return cam_d;
  }
};
