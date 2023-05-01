#pragma once

#include "Eigen/Dense"

using Eigen::Vector3f;
using Eigen::Vector3d;
using Eigen::Vector3i;
using Eigen::Array3d;


RAYTRACER_HOST_DEVICE_FUNC inline float illum(const Array3d &c) {
  return (float) (0.2126f * c.x() + 0.7152f * c.y() + 0.0722f * c.z());
}


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

  RAYTRACER_HOST_DEVICE_FUNC void update_pixel(const Array3d &c, uint x, uint y) const {
    assert(0 <= x && x < w);
    assert(0 <= y && y < h);
    data[x + y * w] = (c.max(0).min(1) * 255).cast<int>();
  }

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
};