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
  float aperture;
  float focal_distance;

  __host__ Camera(const Vector3f &look_from, const Vector3f &look_at, const Vector3f &up,
                  float vFov, float hFov, float nClip, float fClip, float aperture = 0, float focal_distance = 0) {
    this->nClip = nClip;
    this->fClip = fClip;
    this->vFov = vFov;
    this->hFov = hFov;
    this->origin = look_from;
    this->aperture = aperture;
    this->focal_distance = focal_distance;

    Vector3f dirToCamera = look_from - look_at;
    Vector3f screenXDir = up.cross(dirToCamera).normalized();
    Vector3f screenYDir = dirToCamera.cross(screenXDir).normalized();

    c2w.col(0) = screenXDir;
    c2w.col(1) = screenYDir;
    c2w.col(2) = dirToCamera.normalized();
  }

  __device__ __host__ Ray generate_ray(float x, float y) const {
    Vector3f camera_vector = {(0.5f - x) * tanf(hFov * PI / 360) * 2, (0.5f - y) * tanf(vFov * PI / 360) * 2, -1};
    Ray world_ray = Ray(origin, (c2w * camera_vector).normalized());
    world_ray.min_t = nClip;
    world_ray.max_t = fClip;
    return world_ray;
  }

  __device__ __host__ Ray generate_ray_for_thin_lens(float x, float y, float rndR, float rndTheta) const {
    Vector3f camera_vector = {(0.5f - x) * tanf(hFov * PI / 360) * 2, (0.5f - y) * tanf(vFov * PI / 360) * 2, -1};

    Vector3f p_lens = {aperture * sqrtf(rndR) * cosf(rndTheta), aperture * sqrtf(rndR) * sinf(rndTheta), 0};
    Vector3f p_focus = camera_vector * focal_distance;

    Ray focus_ray = Ray((c2w * p_lens) + origin, c2w * (p_focus - p_lens).normalized());
    focus_ray.min_t = nClip;
    focus_ray.max_t = fClip;

    return focus_ray;
  }

  __host__ Camera *to_cuda() const {
    Camera *cam_d;

    cudaMalloc(&cam_d, sizeof(Camera));
    cudaMemcpy(cam_d, this, sizeof(Camera), cudaMemcpyHostToDevice);

    return cam_d;
  }
};
