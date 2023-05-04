#pragma once

#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/linalg/Matrix3f.cuh"
#include "raytracer/scene/ray.cuh"


class Camera {
public:
  Vector3f origin;
  Matrix3f c2w;
  float hFov;
  float vFov;
  float nClip;
  float fClip;
  float aperture;
  float focal_distance;

  Camera(const Vector3f &look_from, const Vector3f &look_at, const Vector3f &up,
         float hFov, float vFov, float nClip, float fClip, float aperture = 0, float focal_distance = 0) {
    this->nClip = nClip;
    this->fClip = fClip;
    this->hFov = hFov;
    this->vFov = vFov;
    this->origin = look_from;
    this->aperture = aperture;
    this->focal_distance = focal_distance;

    Vector3f dirToCamera = look_from - look_at;
    Vector3f screenXDir = up.cross(dirToCamera).unit();
    Vector3f screenYDir = dirToCamera.cross(screenXDir).unit();

    c2w[0] = screenXDir;
    c2w[1] = screenYDir;
    c2w[2] = dirToCamera.unit();
  }

  RAYTRACER_DEVICE_FUNC Ray generate_ray(float x, float y) const {
    Vector3f camera_vector = {(0.5f - x) * tanf(hFov * PI / 360) * 2, (0.5f - y) * tanf(vFov * PI / 360) * 2, -1};
    Ray world_ray = Ray(origin, (c2w * camera_vector).unit());
    world_ray.min_t = nClip;
    world_ray.max_t = fClip;
    return world_ray;
  }

  RAYTRACER_DEVICE_FUNC Ray generate_ray_for_thin_lens(float x, float y, uint *seed) const {
    auto rand = Sampler2D::sample_grid(seed);
    float rndR = rand[0];
    float rndTheta = rand[1] * 2.0 * PI;
    Vector3f camera_vector = {(0.5f - x) * tanf(hFov * PI / 360) * 2, (0.5f - y) * tanf(vFov * PI / 360) * 2, -1};

    Vector3f p_lens = {aperture * sqrtf(rndR) * cosf(rndTheta), aperture * sqrtf(rndR) * sinf(rndTheta), 0};
    Vector3f p_focus = camera_vector * focal_distance;

    Ray focus_ray = Ray((c2w * p_lens) + origin, c2w * (p_focus - p_lens).unit());
    focus_ray.min_t = nClip;
    focus_ray.max_t = fClip;

    return focus_ray;
  }
};
