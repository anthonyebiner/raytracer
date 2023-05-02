#pragma once

#include "raytracer/bvh/bvh.cuh"
#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/scene/lights.cuh"
#include "raytracer/scene/camera.cuh"
#include "raytracer/util/image.cuh"


struct Parameters {
  uint samples_per_pixel = 1024;
  uint samples_per_light = 1;
  uint samples_per_batch = 64;
  float max_tolerance = 0.005f;
  uint max_ray_depth = 32;
  uint bvh_node_size = 4;
} default_parameters;


struct Scene {
  BVHAccelOpt *bvh;

  SceneLight *lights;
  uint num_lights;

  Camera *camera;
};


static RAYTRACER_DEVICE_FUNC Vector3f
estimate_direct_lighting(Scene *scene, Parameters *parameters, const Intersection &isect, uint *seed) {
  if (isect.primitive->bsdf->is_delta()) return {0, 0, 0};
  Vector3f to_light;
  float distanceToLight, pdf;
  Vector3f bump = isect.normal * 0.001;

  Vector3f color = {0, 0, 0};
  for (uint i = 0; i < scene->num_lights; i++) {
    for (uint _ = 0; _ < parameters->samples_per_light; _++) {
      SceneLight light = scene->lights[i];
      Vector3f L = light.sample(isect.hit_point + bump, &to_light, &distanceToLight, &pdf, seed);
      Ray shadow_ray = Ray(isect.hit_point + bump, to_light, distanceToLight - EPS_F);
      if (!scene->bvh->has_intersection(shadow_ray)) {
        Vector3f f = isect.primitive->bsdf->f(isect.o_out, isect.w2o * to_light);
        float cos = fmaxf(isect.normal.dot(shadow_ray.direction.unit()), 0.f);
        color += f * L * cos / pdf / parameters->samples_per_light;
      }
    }
  }
  return color;
}


static RAYTRACER_DEVICE_FUNC Vector3f
estimate_global_lighting(Ray &ray, Scene *scene, Parameters *parameters, uint *seed) {
  Vector3f color_mask = {1, 1, 1};
  Vector3f total_color = {0, 0, 0};
  Intersection prev_isect;

  for (uint i = 0; i <= parameters->max_ray_depth; i++) {
    Intersection isect;
    if (!scene->bvh->intersect(ray, &isect)) break;
    isect.compute();

    if ((i == 0 || prev_isect.primitive->bsdf->is_delta()) && isect.primitive->bsdf->type == BSDF::EMISSION) {
      total_color += color_mask * isect.primitive->bsdf->get_emission();
    }

    if (!isect.primitive->bsdf->is_delta()) {
      total_color += color_mask * estimate_direct_lighting(scene, parameters, isect, seed);
    }

    float cpdf = fmaxf(color_mask[0], fmaxf(color_mask[1], color_mask[2]));
    if (!Sampler1D::coin_flip(seed, cpdf)) {
      break;
    }

    float pdf;
    Vector3f mask;
    Vector3f o_in = isect.primitive->bsdf->sample(isect.o_out, &mask, &pdf, seed);
    if (pdf == 0) break;

    Vector3f bump = isect.normal * 0.001;
    bump = o_in.z > 0 ? bump : -bump;
    ray = Ray(isect.hit_point + bump, isect.o2w * o_in);

    color_mask = color_mask * mask * o_in.z / pdf / fminf(cpdf, 1.f);

    prev_isect = isect;
  }
  return total_color;
}


static RAYTRACER_DEVICE_FUNC void
fill_color(uint x, uint y, Scene *scene, Parameters *parameters, SampleBuffer *buffer, uint *seed) {
  Vector2f origin = Vector2f(x, y);
  Vector3f color = {0, 0, 0};
  uint samples_taken = 1;
  float summed_color = 0;
  float summed_color_squared = 0;
  while (samples_taken <= parameters->samples_per_pixel) {
    auto point = (origin + Sampler2D::sample_grid(seed));
    Ray ray;
    if (scene->camera->aperture < EPS_F) {
      ray = scene->camera->generate_ray(point.x / buffer->w, point.y / buffer->h);
    } else {
      auto rand = Sampler2D::sample_grid(seed);
      ray = scene->camera->generate_ray_for_thin_lens(point.x / buffer->w, point.y / buffer->h, rand[0],
                                                      rand[1] * 2.0 * PI);
    }

    Vector3f sample_color = estimate_global_lighting(ray, scene, parameters, seed);

    color += sample_color;
    summed_color += sample_color.illum();
    summed_color_squared += summed_color * summed_color;
    if (samples_taken % parameters->samples_per_batch == 0) {
      float variance = (summed_color_squared - pow(summed_color, 2) / samples_taken) / (samples_taken - 1);
      float mean = summed_color / samples_taken;
      if (1.96 * pow(variance / samples_taken, 0.5) <= parameters->max_tolerance * mean) {
        break;
      }
    }
    samples_taken++;
  }
  color /= samples_taken - 1;
  buffer->update_pixel(color, x, y);
}