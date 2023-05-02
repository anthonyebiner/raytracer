#pragma once

#include <iostream>
#include <memory>
#include <condition_variable>
#include "raytracer/scene/camera.cuh"
#include "raytracer/util/image.cuh"
#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/util/sampler.cuh"
#include "raytracer/bvh/bvh.cuh"
#include "raytracer/util/misc.cuh"
#include "raytracer/scene/lights.cuh"
#include "raytracer/raytracer.cuh"
#include "fmt/core.h"
#include "raytracer/util/bitmap_image.cuh"

#define USE_PROGRESS true


using fmt::print;
using std::shared_ptr;


void raytrace_cpu(Scene *scene, Parameters *parameters, SampleBuffer *buffer) {
  for (uint y = 0; y < buffer->h; y++) {
    uint x;
    for (x = 0; x < buffer->w; x++) {
      uint seed = 0x42ffffff & (y * buffer->w + x);
      for (uint _ = 0; _ < 10; _++) random_float(&seed);

      fill_color(x, y, scene, parameters, buffer, &seed);
    }
#if USE_PROGRESS
    print_progress((float) (x + y * buffer->w) / (float) (buffer->w * buffer->h));
#endif
  }
}


#ifdef __CUDACC__

__global__ void raytrace_cuda(Scene *scene, Parameters *parameters,
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

  uint seed = 0x42ffffff & (y * buffer->w + x);
  for (uint _ = 0; _ < 10; _++) random_float(&seed);

  fill_color(x, y, scene, parameters, buffer, &seed);

#if USE_PROGRESS
  if (!(threadIdx.x || threadIdx.y)) {
    atomicAdd((int *) progress, 1);
    __threadfence_system();
  }
#endif
}

#endif


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
#ifdef __CUDACC__
    if (d_bvh_opt != nullptr) cudaFree(d_bvh_opt);
#endif
  }

  void resize(uint width, uint height) const {
    sample_buffer->resize(width, height);
  }

  void set_scene(const std::vector<Primitive *> &_primitives, const std::vector<SceneLight> &_lights) {
    primitives.clear();
    primitives = std::vector<Primitive *>(_primitives);
    lights.clear();
    lights = std::vector<SceneLight>(_lights);
#ifdef __CUDACC__
    if (d_bvh_opt != nullptr) cudaFree(d_bvh_opt);
    d_bvh_opt = nullptr;
#endif
  }

  void set_camera(const Vector3f &look_from, const Vector3f &look_at, const Vector3f &up,
                  float hFov, float vFov, float nClip, float fClip, float aperture = 0, float focal_distance = 0) {
    if (focal_distance == 0) {
      focal_distance = (look_from - look_at).norm();
    }
    camera = std::make_shared<Camera>(look_from, look_at, up, hFov, vFov, nClip, fClip, aperture, focal_distance);
  }

  void save_to_file(const std::string &filename) {
    bitmap_image image(sample_buffer->w, sample_buffer->h);
    for (uint x = 0; x < sample_buffer->w; x++) {
      for (uint y = 0; y < sample_buffer->h; y++) {
        Vector3i color = sample_buffer->data[y * sample_buffer->w + x];
        image.set_pixel(x, y, color.x, color.y, color.z);
      }
    }
    image.save_image(filename);
    print("Image saved to {}\n", filename);
  }

#ifdef false

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

    Camera *d_camera;
    cudaMalloc(&d_camera, sizeof(Camera));
    cudaMemcpy(d_camera, camera.get(), sizeof(Camera), cudaMemcpyHostToDevice);
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
    raytrace_cuda<<<blocks, threads>>>(d_scene, d_parameters, d_buf, d_progress);
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

#else

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

    Scene scene = {&bvh_opt, &lights[0], (uint) lights.size(), camera.get()};

    print("Raytracing Pixels\n");
    auto start = std::chrono::steady_clock::now();
    raytrace_cpu(&scene, &parameters, sample_buffer.get());
    auto end = std::chrono::steady_clock::now();
    print("\nPixels Raytraced. Elapsed time = {} ms\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
  }

#endif
};
