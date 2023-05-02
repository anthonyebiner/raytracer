#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include "raytracer/util/misc.cuh"

/**
 * Defines 3D vectors.
 */
class Vector3i {
public:

  union {
    struct {
      int x, y, z;
    };
    struct {
      int r, g, b;
    };
  };

  /**
   * Constructor.
   * Initializes tp vector (0,0,0).
   */
  RAYTRACER_DEVICE_FUNC Vector3i() : x(0), y(0), z(0) {}

  /**
   * Constructor.
   * Initializes to vector (x,y,z).
   */
  RAYTRACER_DEVICE_FUNC Vector3i(int x, int y, int z) : x(x), y(y), z(z) {}

  /**
   * Constructor.
   * Initializes to vector (c,c,c)
   */
  RAYTRACER_DEVICE_FUNC Vector3i(int c) : x(c), y(c), z(c) {}

  /**
   * Constructor.
   * Initializes from existing vector
   */
  RAYTRACER_DEVICE_FUNC Vector3i(const Vector3i &v) : x(v.x), y(v.y), z(v.z) {}

  // dot product (a.k.a. inner or scalar product)
  RAYTRACER_DEVICE_FUNC inline int dot(const Vector3i &v) {
    return v.x * x + v.y * y + v.z * z;
  }

// cross product
  RAYTRACER_DEVICE_FUNC inline Vector3i cross(const Vector3i &v) {
    return Vector3i(y * v.z - z * v.y,
                    z * v.x - x * v.z,
                    x * v.y - y * v.x);
  }

  // returns reference to the specified component (0-based indexing: x, y, z)
  RAYTRACER_DEVICE_FUNC inline int &operator[](const int &index) {
    return (&x)[index];
  }

  // returns const reference to the specified component (0-based indexing: x, y, z)
  RAYTRACER_DEVICE_FUNC inline const int &operator[](const int &index) const {
    return (&x)[index];
  }

  RAYTRACER_DEVICE_FUNC inline bool operator==(const Vector3i &v) const {
    return v.x == x && v.y == y && v.z == z;
  }

  // negation
  RAYTRACER_DEVICE_FUNC inline Vector3i operator-(void) const {
    return Vector3i(-x, -y, -z);
  }

  // addition
  RAYTRACER_DEVICE_FUNC inline Vector3i operator+(const Vector3i &v) const {
    return Vector3i(x + v.x, y + v.y, z + v.z);
  }

  // subtraction
  RAYTRACER_DEVICE_FUNC inline Vector3i operator-(const Vector3i &v) const {
    return Vector3i(x - v.x, y - v.y, z - v.z);
  }

  // element wise multiplication
  RAYTRACER_DEVICE_FUNC inline Vector3i operator*(const Vector3i &v) const {
    return Vector3i(x * v.x, y * v.y, z * v.z);
  }

  // element wise division
  RAYTRACER_DEVICE_FUNC inline Vector3i operator/(const Vector3i &v) const {
    return Vector3i(x / v.x, y / v.y, z / v.z);
  }

  // right scalar multiplication
  RAYTRACER_DEVICE_FUNC inline Vector3i operator*(const int &c) const {
    return Vector3i(x * c, y * c, z * c);
  }

  // scalar division
  RAYTRACER_DEVICE_FUNC inline Vector3i operator/(const int &c) const {
    const int rc = 1.0 / c;
    return Vector3i(rc * x, rc * y, rc * z);
  }

  // addition / assignment
  RAYTRACER_DEVICE_FUNC inline void operator+=(const Vector3i &v) {
    x += v.x;
    y += v.y;
    z += v.z;
  }

  // subtraction / assignment
  RAYTRACER_DEVICE_FUNC inline void operator-=(const Vector3i &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
  }

  // scalar multiplication / assignment
  RAYTRACER_DEVICE_FUNC inline void operator*=(const int &c) {
    x *= c;
    y *= c;
    z *= c;
  }

  // scalar division / assignment
  RAYTRACER_DEVICE_FUNC inline void operator/=(const int &c) {
    (*this) *= (1. / c);
  }

  /**
   * Returns per entry reciprocal
   */
  RAYTRACER_DEVICE_FUNC inline Vector3i rcp(void) const {
    return Vector3i(1.0 / x, 1.0 / y, 1.0 / z);
  }

  /**
   * Returns Euclidean length.
   */
  RAYTRACER_DEVICE_FUNC inline int norm(void) const {
    return sqrt(x * x + y * y + z * z);
  }

  /**
   * Returns Euclidean length squared.
   */
  RAYTRACER_DEVICE_FUNC inline int norm2(void) const {
    return x * x + y * y + z * z;
  }

  /**
   * Returns unit vector.
   */
  RAYTRACER_DEVICE_FUNC inline Vector3i unit(void) const {
    int rNorm = 1. / norm();
    return (*this) * rNorm;
  }

  /**
   * Divides by Euclidean length.
   */
  RAYTRACER_DEVICE_FUNC inline void normalize(void) {
    (*this) /= norm();
  }

  RAYTRACER_DEVICE_FUNC inline int illum() const {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
  }

}; // class Vector3D

// left scalar multiplication
RAYTRACER_DEVICE_FUNC inline Vector3i operator*(const int &c, const Vector3i &v) {
  return Vector3i(c * v.x, c * v.y, c * v.z);
}

// left scalar divide
RAYTRACER_DEVICE_FUNC inline Vector3i operator/(const int &c, const Vector3i &v) {
  return Vector3i(c / v.x, c / v.y, c / v.z);
}
