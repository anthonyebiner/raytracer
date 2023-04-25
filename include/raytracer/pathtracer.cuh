#pragma once

#include <iostream>
#include <memory>
#include <curand_kernel.h>
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
using std::shared_ptr;


__device__ Vector3f get_color(Ray &ray, BVHAccelOpt *bvh, SceneLight *lights, uint num_lights,
                              uint samples_per_light, uint max_ray_depth, curandState rand_state) {
  Vector3f color_mask = Vector3f(1, 1, 1);
  Vector3f color = Vector3f(0, 0, 0);
  for (uint i = 0; i <= max_ray_depth; i++) {
    if (!bvh->intersect(&ray)) break;

    Matrix3f o2w;
    make_coord_space(o2w, ray.hit_n);
    Matrix3f w2o = o2w.transpose();

    Vector3f hit_point = ray.origin + ray.direction * ray.hit_t;
    Primitive *hit_primitive = ray.hit_p;
    Vector3f o_out = w2o * (-ray.direction);

    color += Vector3f(hit_primitive->bsdf->get_emission().array() * color_mask.array());

    Vector3f o_in;
    float distanceToLight;
    Vector3f this_color = Vector3f(0, 0, 0);

    for (uint j = 0; j < num_lights; j++) {
      if (lights[j].is_delta_light()) samples_per_light = 1;
      for (int sample = 0; sample < samples_per_light; sample++) {
        auto L = lights[j].sample(hit_point, &o_in, &distanceToLight, &rand_state);
        Ray shadow_ray = Ray(hit_point, o_in);
        shadow_ray.hit_t = distanceToLight - EPS_F;
        if (!bvh->intersect(&shadow_ray)) {
          Vector3f mask = hit_primitive->bsdf->f(shadow_ray.direction.normalized(), o_out.normalized());
          float cos = BSDF::cos_theta((w2o * o_in).normalized());
          if (cos > 0) this_color += Vector3f(mask.array() * L.array() * cos);
        }
      }
      this_color /= samples_per_light;
    }
    color += Vector3f(this_color.array() * color_mask.array());

    float cpdf = color_mask.maxCoeff();
    if (!CudaSampler1D::coin_flip(&rand_state, cpdf)) {
      break;
    }

    float pdf;
    Vector3f mask = hit_primitive->bsdf->sample(o_out, &o_in, &pdf, &rand_state);
    float cos_theta = BSDF::cos_theta(o_in);
    color_mask = Vector3f(color_mask.array() * mask.array()) * cos_theta / pdf / min(cpdf, 1.);

    ray = Ray(hit_point, o2w * o_in);
  }

  return color.cast<float>();
}


__global__ void raytrace_pixel(Camera *camera, BVHAccelOpt *bvh, SampleBuffer *buffer, curandState *rand_states,
                               SceneLight *lights, uint num_lights, uint samples_per_pixel, uint samples_per_light,
                               uint samples_per_batch, float max_tolerance, uint max_ray_depth,
                               volatile int *progress) {
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
  uint pixel_index = y * buffer->w + x;
  curandState rand_state = rand_states[pixel_index];

  Vector2f origin = Vector2f(x, y);
  Vector3f color = Vector3f::Zero();
  uint samples_taken = 1;
  float summed_color = 0;
  float summed_color_squared = 0;
  while (samples_taken <= samples_per_pixel) {
    auto point = (origin + CudaSampler2D::sample_grid(&rand_state));
    auto ray = camera->generate_ray(point.x() / buffer->w, point.y() / buffer->h);

    Vector3f sample_color = get_color(ray, bvh, lights, num_lights, samples_per_light, max_ray_depth, rand_state);

    color += sample_color;
    summed_color += illum(sample_color);
    summed_color_squared += summed_color * summed_color;
    if (samples_taken % samples_per_batch == 0) {
      float variance = (summed_color_squared - pow(summed_color, 2) / samples_taken) / (samples_taken - 1);
      float mean = summed_color / samples_taken;
      if (1.96 * pow(variance / samples_taken, 0.5) <= max_tolerance * mean) {
        break;
      }
    }
    samples_taken++;
  }
  color /= samples_taken - 1;

  buffer->update_pixel(color, x, y);

#if USE_PROGRESS
  if (!(threadIdx.x || threadIdx.y)) {
    atomicAdd((int *) progress, 1);
    __threadfence_system();
  }
#endif
}


__global__ void init_renderer(SampleBuffer *buffer, curandState *rand_states) {
  uint x = threadIdx.x + blockIdx.x * blockDim.x;
  uint y = threadIdx.y + blockIdx.y * blockDim.y;
  if ((x >= buffer->w) || (y >= buffer->h)) return;
  uint pixel_index = y * buffer->w + x;
  curand_init(1337, pixel_index, 0, &rand_states[pixel_index]);
}


class PathTracer {
public:
  // Parameters //
  uint max_ray_depth;
  uint samples_per_pixel;
  uint samples_per_light;
  uint samples_per_batch;
  float max_tolerance;

  // Components //
  std::vector<Primitive *> primitives;
  std::vector<SceneLight> lights;
  shared_ptr<SampleBuffer> sample_buffer;
  shared_ptr<Camera> camera;

  explicit PathTracer(uint samples_per_pixel = 512, uint samples_per_light = 1, uint samples_per_batch = 32,
                      float max_tolerance = 0.005f, uint max_ray_depth = 32) :
      samples_per_pixel(samples_per_pixel), samples_per_light(samples_per_light), samples_per_batch(samples_per_batch),
      max_tolerance(max_tolerance), max_ray_depth(max_ray_depth), camera(nullptr) {
    sample_buffer = std::make_shared<SampleBuffer>();
  }

  void resize(uint width, uint height) const {
    sample_buffer->resize(width, height);
  }

  void set_scene(const std::vector<Primitive *> &_primitives, const std::vector<SceneLight> &_lights) {
    primitives.clear();
    primitives = std::vector<Primitive *>(_primitives);
    lights.clear();
    lights = std::vector<SceneLight>(_lights);
  }

  void set_camera(const Vector3f &look_from, const Vector3f &look_at, const Vector3f &up,
                  float vFov, float hFov, float nClip, float fClip) {
    camera = std::make_shared<Camera>(look_from, look_at, up, vFov, hFov, nClip, fClip);
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

  void raytrace() const {
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

    BVHAccel bvh = BVHAccel(primitives, 4);
    print("Created BVH. Depth: {}\n", bvh.get_max_depth());

    BVHAccelOpt bvh_opt = bvh.to_optimized_bvh();
    print("Optimized BVH\n");

    BVHAccelOpt *bvh_opt_d = bvh_opt.to_cuda();
    print("Moved BVH to CUDA\n");

    SceneLight *d_lights;
    uint num_lights = lights.size();
    cudaMalloc(&d_lights, sizeof(SceneLight) * num_lights);
    cudaMemcpy(d_lights, &lights[0], sizeof(SceneLight) * lights.size(), cudaMemcpyHostToDevice);
    print("Moved lights to CUDA\n");

    SampleBuffer *buf_d = sample_buffer->to_cuda();
    print("Moved sample buffer to CUDA\n");

    Camera *camera_d = camera->to_cuda();
    print("Moved camera to CUDA\n");

    curandState *rand_states_d;
    cudaMalloc((curandState **) &rand_states_d, sample_buffer->w * sample_buffer->h * sizeof(curandState));

    dim3 blocks(sample_buffer->w / 16 + 1, sample_buffer->h / 16 + 1);
    dim3 threads(16, 16);
    init_renderer<<<blocks, threads>>>(buf_d, rand_states_d);
    cudaCheckErrors("init_renderer Error");
    cudaDeviceSynchronize();
    print("CUDA renderer initialized\n");

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
    raytrace_pixel<<<blocks, threads>>>(camera_d, bvh_opt_d, buf_d, rand_states_d, d_lights, num_lights,
                                        samples_per_pixel, samples_per_light,
                                        samples_per_batch, max_tolerance, max_ray_depth, d_progress);
    cudaEventRecord(stop);
#if USE_PROGRESS
    uint num_blocks = blocks.x * blocks.y;
    float percentage;
    do {
      percentage = (float) *h_progress / (float) num_blocks;
      print_progress((float) *h_progress / (float) num_blocks);
    } while (percentage < 0.95f);
    print_progress(1.0);
    printf("\n");
#endif
    cudaEventSynchronize(stop);
    cudaCheckErrors("cudaEventSynchronize error");
    float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaCheckErrors("cudaEventElapsedTime error");
    cudaDeviceSynchronize();
    cudaCheckErrors("raytrace_pixel error");

    print("Pixels Raytraced. Elapsed time = {} ms\n", et);

    sample_buffer->from_cuda(buf_d);
    print("Sample buffer loaded from CUDA\n");
  }
};
