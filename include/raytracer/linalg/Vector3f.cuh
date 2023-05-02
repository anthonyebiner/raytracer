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
class Vector3f {
public:

  union {
    struct {
      float x, y, z;
    };
    struct {
      float r, g, b;
    };
  };

  /**
   * Constructor.
   * Initializes tp vector (0,0,0).
   */
  RAYTRACER_DEVICE_FUNC Vector3f() : x(0.0), y(0.0), z(0.0) {}

  /**
   * Constructor.
   * Initializes to vector (x,y,z).
   */
  RAYTRACER_DEVICE_FUNC Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}

  /**
   * Constructor.
   * Initializes to vector (c,c,c)
   */
  RAYTRACER_DEVICE_FUNC Vector3f(float c) : x(c), y(c), z(c) {}

  /**
   * Constructor.
   * Initializes from existing vector
   */
  RAYTRACER_DEVICE_FUNC Vector3f(const Vector3f &v) : x(v.x), y(v.y), z(v.z) {}

  // dot product (a.k.a. inner or scalar product)
  RAYTRACER_DEVICE_FUNC inline float dot(const Vector3f &v) const {
    return v.x * x + v.y * y + v.z * z;
  }

// cross product
  RAYTRACER_DEVICE_FUNC inline Vector3f cross(const Vector3f &v) const {
    return Vector3f(y * v.z - z * v.y,
                    z * v.x - x * v.z,
                    x * v.y - y * v.x);
  }

  // returns reference to the specified component (0-based indexing: x, y, z)
  RAYTRACER_DEVICE_FUNC inline float &operator[](const int &index) {
    return (&x)[index];
  }

  // returns const reference to the specified component (0-based indexing: x, y, z)
  RAYTRACER_DEVICE_FUNC inline const float &operator[](const int &index) const {
    return (&x)[index];
  }

  RAYTRACER_DEVICE_FUNC inline bool operator==(const Vector3f &v) const {
    return v.x == x && v.y == y && v.z == z;
  }

  // negation
  RAYTRACER_DEVICE_FUNC inline Vector3f operator-(void) const {
    return Vector3f(-x, -y, -z);
  }

  // addition
  RAYTRACER_DEVICE_FUNC inline Vector3f operator+(const Vector3f &v) const {
    return Vector3f(x + v.x, y + v.y, z + v.z);
  }

  // subtraction
  RAYTRACER_DEVICE_FUNC inline Vector3f operator-(const Vector3f &v) const {
    return Vector3f(x - v.x, y - v.y, z - v.z);
  }

  // element wise multiplication
  RAYTRACER_DEVICE_FUNC inline Vector3f operator*(const Vector3f &v) const {
    return Vector3f(x * v.x, y * v.y, z * v.z);
  }

  // element wise division
  RAYTRACER_DEVICE_FUNC inline Vector3f operator/(const Vector3f &v) const {
    return Vector3f(x / v.x, y / v.y, z / v.z);
  }

  // right scalar multiplication
  RAYTRACER_DEVICE_FUNC inline Vector3f operator*(const float &c) const {
    return Vector3f(x * c, y * c, z * c);
  }

  // scalar division
  RAYTRACER_DEVICE_FUNC inline Vector3f operator/(const float &c) const {
    const float rc = 1.0 / c;
    return Vector3f(rc * x, rc * y, rc * z);
  }

  // addition / assignment
  RAYTRACER_DEVICE_FUNC inline void operator+=(const Vector3f &v) {
    x += v.x;
    y += v.y;
    z += v.z;
  }

  // subtraction / assignment
  RAYTRACER_DEVICE_FUNC inline void operator-=(const Vector3f &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
  }

  // scalar multiplication / assignment
  RAYTRACER_DEVICE_FUNC inline void operator*=(const float &c) {
    x *= c;
    y *= c;
    z *= c;
  }

  // scalar division / assignment
  RAYTRACER_DEVICE_FUNC inline void operator/=(const float &c) {
    (*this) *= (1. / c);
  }

  /**
   * Returns per entry reciprocal
   */
  RAYTRACER_DEVICE_FUNC inline Vector3f rcp(void) const {
    return Vector3f(1.0 / x, 1.0 / y, 1.0 / z);
  }

  /**
   * Returns Euclidean length.
   */
  RAYTRACER_DEVICE_FUNC inline float norm(void) const {
    return sqrt(x * x + y * y + z * z);
  }

  /**
   * Returns Euclidean length squared.
   */
  RAYTRACER_DEVICE_FUNC inline float norm2(void) const {
    return x * x + y * y + z * z;
  }

  /**
   * Returns unit vector.
   */
  RAYTRACER_DEVICE_FUNC inline Vector3f unit(void) const {
    float rNorm = 1. / norm();
    return (*this) * rNorm;
  }

  /**
   * Divides by Euclidean length.
   */
  RAYTRACER_DEVICE_FUNC inline void normalize(void) {
    (*this) /= norm();
  }

  RAYTRACER_DEVICE_FUNC inline float illum() const {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
  }

}; // class Vector3D

// left scalar multiplication
RAYTRACER_DEVICE_FUNC inline Vector3f operator*(const float &c, const Vector3f &v) {
  return Vector3f(c * v.x, c * v.y, c * v.z);
}

// left scalar divide
RAYTRACER_DEVICE_FUNC inline Vector3f operator/(const float &c, const Vector3f &v) {
  return Vector3f(c / v.x, c / v.y, c / v.z);
}
