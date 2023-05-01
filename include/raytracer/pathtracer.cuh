#pragma once

#include <iostream>
#include <memory>
#include <curand_kernel.h>
#include <condition_variable>
#include "raytracer/scene/camera.cuh"
#include "raytracer/util/image.cuh"
#include "raytracer/util/sampler.cuh"
#include "raytracer/bvh/bvh.cuh"
#include "raytracer/util/misc.cuh"
#include "raytracer/scene/lights.cuh"
#include "Eigen/Dense"
#include "fmt/core.h"
#include "raytracer/util/bitmap_image.cuh"

#define USE_PROGRESS true


using fmt::print;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Array3f;
using Eigen::Array3d;
using std::shared_ptr;


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


__device__ __host__ Array3d
estimate_direct_lighting(Scene *scene, Parameters *parameters, Intersection isect, uint *seed) {
  if (isect.primitive->bsdf->is_delta()) return {0, 0, 0};
  Vector3f to_light;
  float distanceToLight, pdf;
  Intersection shadow_isect;
  Vector3f bump = isect.normal * 0.001;

  Array3d color = {0, 0, 0};
  for (uint i = 0; i < scene->num_lights; i++) {
    SceneLight light = scene->lights[i];
    Array3f L = light.sample(isect.hit_point + bump, &to_light, &distanceToLight, &pdf, seed);
    Ray shadow_ray = Ray(isect.hit_point + bump, to_light, distanceToLight - EPS_F);
    if (!scene->bvh->intersect(&shadow_ray, &shadow_isect)) {
      Array3f f = isect.primitive->bsdf->f(isect.o_out, isect.w2o * to_light);
      float cos = max(isect.normal.dot(shadow_ray.direction.normalized()), 0.f);
      color += f.cast<double>() * L.cast<double>() * cos / pdf;
    }
  }
  return color;
}


__device__ __host__ Array3d
estimate_global_lighting(Ray &ray, Scene *scene, Parameters *parameters, uint *seed) {
  Array3d color_mask = {1, 1, 1};
  Array3d total_color = {0, 0, 0};
  Intersection prev_isect;

  for (uint i = 0; i <= parameters->max_ray_depth; i++) {
    Intersection isect;
    if (!scene->bvh->intersect(&ray, &isect)) break;
    isect.compute();

    if ((i == 0 || prev_isect.primitive->bsdf->is_delta()) && isect.primitive->bsdf->type == BSDF::EMISSION) {
      total_color += color_mask * isect.primitive->bsdf->get_emission().array().cast<double>();
    }

    if (!isect.primitive->bsdf->is_delta()) {
      total_color += color_mask * estimate_direct_lighting(scene, parameters, isect, seed);
    }

    float cpdf = color_mask.maxCoeff();
    if (!Sampler1D::coin_flip(seed, cpdf)) {
      break;
    }

    float pdf;
    Array3f mask;
    Vector3f o_in = isect.primitive->bsdf->sample(isect.o_out, &mask, &pdf, seed);
    if (pdf == 0) break;

    Vector3f bump = isect.normal * 0.001;
    bump = o_in.z() > 0 ? bump : -bump;
    ray = Ray(isect.hit_point + bump, isect.o2w * o_in);

    color_mask *= mask.cast<double>() * o_in.z() / pdf / fminf(cpdf, 1.f);

    prev_isect = isect;
  }
  return total_color;
}


__device__ __host__ Array3d
fill_color(uint x, uint y, Scene *scene, Parameters *parameters, SampleBuffer *buffer, uint *seed) {
  Vector2f origin = Vector2f(x, y);
  Array3d color = {0, 0, 0};
  uint samples_taken = 1;
  float summed_color = 0;
  float summed_color_squared = 0;
  while (samples_taken <= parameters->samples_per_pixel) {
    auto point = (origin + Sampler2D::sample_grid(seed));
    Ray ray;
    if (scene->camera->aperture < EPS_F) {
      ray = scene->camera->generate_ray(point.x() / buffer->w, point.y() / buffer->h);
    } else {
      auto rand = Sampler2D::sample_grid(seed);
      ray = scene->camera->generate_ray_for_thin_lens(point.x() / buffer->w, point.y() / buffer->h, rand[0],
                                                      rand[1] * 2.0 * PI);
    }

    Array3d sample_color = estimate_global_lighting(ray, scene, parameters, seed);

    color += sample_color;
    summed_color += illum(sample_color);
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

__host__ void raytrace_cpu(Scene *scene, Parameters *parameters, SampleBuffer *buffer) {
  for (uint y = 0; y < buffer->h; y++) {
    for (uint x = 0; x < buffer->w; x++) {
      uint seed = y * buffer->w + x;
      Sampler3D::sample_grid(&seed);
      fill_color(x, y, scene, parameters, buffer, &seed);
#if USE_PROGRESS
      print_progress((float) (x + y * buffer->w) / (float) (buffer->w * buffer->h));
#endif
    }
  }
}

__global__ void raytrace_kernel(Scene *scene, Parameters *parameters,
                                SampleBuffer *buffer, volatile int *progress) {
  uint x = threadIdx.x + blockIdx.x * blockDim.x;
  uint y = threadIdx.y + blockIdx.y * blockDim.y;

  if ((x >= buffer->w) || (y >= buffer->h)) {
#if USE_PROGRESS
    if (!(threadIdx.x || threadIdx.y)) {
      atomicAdd((int *) progress, 1);
      __threadfence_system();
    }
#endif
    return;
  }

  uint seed = y * buffer->w + x;
  Sampler3D::sample_grid(&seed);
  fill_color(x, y, scene, parameters, buffer, &seed);

#if USE_PROGRESS
  if (!(threadIdx.x || threadIdx.y)) {
    atomicAdd((int *) progress, 1);
    __threadfence_system();
  }
#endif
}


class PathTracer {
public:
  BVHAccelOpt *d_bvh_opt;

  Parameters parameters;
  std::vector<Primitive *> primitives;
  std::vector<SceneLight> lights;
  shared_ptr<Camera> camera;
  shared_ptr<SampleBuffer> sample_buffer;

  explicit PathTracer(Parameters parameters = default_parameters) :
      camera(nullptr), parameters(parameters), d_bvh_opt(nullptr) {
    sample_buffer = std::make_shared<SampleBuffer>();
  }

  ~PathTracer() {
    if (d_bvh_opt != nullptr) cudaFree(d_bvh_opt);
  }

  void resize(uint width, uint height) const {
    sample_buffer->resize(width, height);
  }

  void set_scene(const std::vector<Primitive *> &_primitives, const std::vector<SceneLight> &_lights) {
    primitives.clear();
    primitives = std::vector<Primitive *>(_primitives);
    lights.clear();
    lights = std::vector<SceneLight>(_lights);
    if (d_bvh_opt != nullptr) cudaFree(d_bvh_opt);
    d_bvh_opt = nullptr;
  }

  void set_camera(const Vector3f &look_from, const Vector3f &look_at, const Vector3f &up,
                  float vFov, float hFov, float nClip, float fClip, float aperture = 0, float focal_distance = 0) {
    if (focal_distance == 0) {
      focal_distance = (look_from - look_at).norm();
    }
    camera = std::make_shared<Camera>(look_from, look_at, up, vFov, hFov, nClip, fClip, aperture, focal_distance);
  }

  void save_to_file(const std::string &filename) {
    bitmap_image image(sample_buffer->w, sample_buffer->h);
    for (uint x = 0; x < sample_buffer->w; x++) {
      for (uint y = 0; y < sample_buffer->h; y++) {
        Vector3i color = sample_buffer->data[y * sample_buffer->w + x];
        image.set_pixel(x, y, color.x(), color.y(), color.z());
      }
    }
    image.save_image(filename);
    print("Image saved to {}\n", filename);
  }


  void raytrace() {
    if (!camera) {
      print("CAMERA NOT SET!");
      return;
    }
    if (primitives.empty()) {
      print("SCENE NOT SET!");
      return;
    }
    if (lights.empty()) {
      print("LIGHTS NOT SET!");
      return;
    }
    if (sample_buffer->w == 0 || sample_buffer->h == 0) {
      print("SAMPLE BUFFER EMPTY");
      return;
    }
    print("Starting raytracing...\n");

    print("Creating BVH\n");
    BVHAccel bvh = BVHAccel(primitives, parameters.bvh_node_size);
    print("Created BVH. Depth: {}\n", bvh.get_max_depth());

    BVHAccelOpt bvh_opt = bvh.to_optimized_bvh();
    print("Optimized BVH\n");

    Scene scene = {&bvh_opt, &lights[0], uint(lights.size()), camera.get()};

    print("Raytracing Pixels\n");
    auto start = std::chrono::steady_clock::now();
    raytrace_cpu(&scene, &parameters, sample_buffer.get());
    auto end = std::chrono::steady_clock::now();
    print("Pixels Raytraced. Elapsed time = {} ms\n",
          std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  }

  void raytrace_cuda() {
    if (!camera) {
      print("CAMERA NOT SET!");
      return;
    }
    if (primitives.empty()) {
      print("SCENE NOT SET!");
      return;
    }
    if (lights.empty()) {
      print("LIGHTS NOT SET!");
      return;
    }
    if (sample_buffer->w == 0 || sample_buffer->h == 0) {
      print("SAMPLE BUFFER EMPTY");
      return;
    }
    print("Starting raytracing...\n");

    if (d_bvh_opt == nullptr) {
      print("Creating BVH\n");
      BVHAccel bvh = BVHAccel(primitives, parameters.bvh_node_size);
      print("Created BVH. Depth: {}\n", bvh.get_max_depth());

      BVHAccelOpt bvh_opt = bvh.to_optimized_bvh();
      print("Optimized BVH\n");

      d_bvh_opt = bvh_opt.to_cuda();
      print("Moved BVH to CUDA\n");
    }

    uint num_lights = lights.size();
    SceneLight *d_lights;
    cudaMalloc(&d_lights, sizeof(SceneLight) * num_lights);
    cudaMemcpy(d_lights, &lights[0], sizeof(SceneLight) * lights.size(), cudaMemcpyHostToDevice);
    print("Moved lights to CUDA\n");

    SampleBuffer *d_buf = sample_buffer->to_cuda();
    print("Moved sample buffer to CUDA\n");

    Camera *d_camera = camera->to_cuda();
    print("Moved camera to CUDA\n");

    Scene *d_scene;
    Scene scene = {d_bvh_opt, d_lights, num_lights, d_camera};
    cudaMalloc(&d_scene, sizeof(Scene));
    cudaMemcpy(d_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice);
    print("Moved scene to CUDA\n");

    Parameters *d_parameters;
    cudaMalloc(&d_parameters, sizeof(Parameters));
    cudaMemcpy(d_parameters, &parameters, sizeof(Parameters), cudaMemcpyHostToDevice);
    print("Moved parameters to CUDA\n");

    dim3 blocks(sample_buffer->w / 16 + 1, sample_buffer->h / 16 + 1);
    dim3 threads(16, 16);

    volatile int *d_progress, *h_progress;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaCheckErrors("cudaSetDeviceFlags error");
    cudaHostAlloc((void **) &h_progress, sizeof(int), cudaHostAllocMapped);
    cudaCheckErrors("cudaHostAlloc error");
    cudaHostGetDevicePointer((int **) &d_progress, (int *) h_progress, 0);
    cudaCheckErrors("cudaHostGetDevicePointer error");
    *h_progress = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    print("Raytracing Pixels\n");
    cudaEventRecord(start);
    raytrace_kernel<<<blocks, threads>>>(d_scene, d_parameters, d_buf, d_progress);
    cudaEventRecord(stop);
#if USE_PROGRESS
    uint num_blocks = blocks.x * blocks.y;
    float percentage;
    do {
      percentage = (float) *h_progress / (float) num_blocks;
      print_progress(percentage);
    } while (percentage < 0.98f);
    print_progress(1.0);
    printf("\n");
#endif
    cudaEventSynchronize(stop);
    cudaCheckErrors("cudaEventSynchronize error");
    float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaCheckErrors("cudaEventElapsedTime error");
    cudaDeviceSynchronize();
    cudaCheckErrors("raytrace_kernel error");

    print("Pixels Raytraced. Elapsed time = {} ms\n", et);

    sample_buffer->from_cuda(d_buf);
    print("Sample buffer loaded from CUDA\n");

    cudaFree(d_buf);
    cudaFree(d_camera);
    cudaFree(d_scene);
    cudaFree(d_parameters);
  }
};
