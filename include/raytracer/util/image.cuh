#pragma once

#include <cassert>
#include "raytracer/linalg/Vector3i.cuh"
#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/util/misc.cuh"


class SampleBuffer {
public:
  uint w;
  uint h;
  Vector3i *data;

  SampleBuffer() : w(0), h(0), data(nullptr) {}

  SampleBuffer(uint w, uint h) : w(w), h(h) {
    data = (Vector3i *) malloc(w * h * sizeof(Vector3i));
    clear();
  }

  void resize(uint new_w, uint new_h) {
    w = new_w;
    h = new_h;
    delete[] data;
    data = (Vector3i *) malloc(w * h * sizeof(Vector3i));
    clear();
  }

  void clear() const {
    for (uint i = 0; i < w * h; ++i) {
      data[i] = Vector3i(0, 0, 0);
    }
  }

  RAYTRACER_DEVICE_FUNC void update_pixel(const Vector3f &c, uint x, uint y) const {
    assert(0 <= x && x < w);
    assert(0 <= y && y < h);
    float gamma = 2.2f;
    float level = 1.0f;
    float one_over_gamma = 1.0f / gamma;
    float exposure = sqrt(pow(2, level));
    data[x + y * w] = Vector3i(
        255 * maxf(0.f, minf(1.f, powf(c.r * exposure, one_over_gamma))),
        255 * maxf(0.f, minf(1.f, powf(c.g * exposure, one_over_gamma))),
        255 * maxf(0.f, minf(1.f, powf(c.b * exposure, one_over_gamma)))
    );
  }

#ifdef __CUDACC__

  SampleBuffer *to_cuda() const {
    SampleBuffer *buffer;

    cudaMalloc(&buffer, sizeof(SampleBuffer));
    cudaMemcpy(buffer, this, sizeof(SampleBuffer), cudaMemcpyHostToDevice);

    Vector3f *d_data;
    cudaMalloc(&d_data, sizeof(Vector3f) * w * h);
    cudaMemcpy(d_data, data, sizeof(Vector3f) * w * h, cudaMemcpyHostToDevice);
    cudaMemcpy(&(buffer->data), &d_data, sizeof(Vector3f *), cudaMemcpyHostToDevice);

    return buffer;
  }

  void from_cuda(SampleBuffer *other) {
    delete[] data;
    cudaMemcpy(this, other, sizeof(SampleBuffer), cudaMemcpyDeviceToHost);

    Vector3i *d_data = data;
    data = (Vector3i *) malloc(sizeof(Vector3i) * w * h);
    cudaMemcpy(data, d_data, sizeof(Vector3i) * w * h, cudaMemcpyDeviceToHost);
  }

#endif
};