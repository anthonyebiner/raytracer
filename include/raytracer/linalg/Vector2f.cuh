#pragma once

#include <cmath>
#include "raytracer/util/misc.cuh"

/**
 * Defines 2D vectors.
 */
class Vector2f {
public:

  // components
  float x, y;

  /**
   * Constructor.
   * Initializes to vector (0,0).
   */
  RAYTRACER_DEVICE_FUNC Vector2f() : x(0.0), y(0.0) {}

  /**
   * Constructor.
   * Initializes to vector (a,b).
   */
  RAYTRACER_DEVICE_FUNC Vector2f(float x, float y) : x(x), y(y) {}

  /**
   * Constructor.
   * Copy constructor. Creates a copy of the given vector.
   */
  RAYTRACER_DEVICE_FUNC Vector2f(const Vector2f &v) : x(v.x), y(v.y) {}

  // inner product
  RAYTRACER_DEVICE_FUNC inline const float dot(const Vector2f &v) {
    return x * v.x + y * v.y;
  }

  // cross product
  RAYTRACER_DEVICE_FUNC inline const float cross(const Vector2f &v) {
    return x * v.y - y * v.x;
  }

  // returns reference to the specified component (0-based indexing: x, y)
  RAYTRACER_DEVICE_FUNC inline float &operator[](const int &index) {
    return (&x)[index];
  }

  // returns const reference to the specified component (0-based indexing: x, y)
  RAYTRACER_DEVICE_FUNC inline const float &operator[](const int &index) const {
    return (&x)[index];
  }

  // additive inverse
  RAYTRACER_DEVICE_FUNC inline const Vector2f operator-(void) const {
    return Vector2f(-x, -y);
  }

  // addition
  RAYTRACER_DEVICE_FUNC inline const Vector2f operator+(const Vector2f &v) const {
    Vector2f u = *this;
    u += v;
    return u;
  }

  // subtraction
  RAYTRACER_DEVICE_FUNC inline const Vector2f operator-(const Vector2f &v) const {
    Vector2f u = *this;
    u -= v;
    return u;
  }

  // right scalar multiplication
  RAYTRACER_DEVICE_FUNC inline const Vector2f operator*(float r) const {
    Vector2f vr = *this;
    vr *= r;
    return vr;
  }

  // scalar division
  RAYTRACER_DEVICE_FUNC inline const Vector2f operator/(float r) const {
    Vector2f vr = *this;
    vr /= r;
    return vr;
  }

  // add v
  RAYTRACER_DEVICE_FUNC inline void operator+=(const Vector2f &v) {
    x += v.x;
    y += v.y;
  }

  // subtract v
  RAYTRACER_DEVICE_FUNC inline void operator-=(const Vector2f &v) {
    x -= v.x;
    y -= v.y;
  }

  // scalar multiply by r
  RAYTRACER_DEVICE_FUNC inline void operator*=(float r) {
    x *= r;
    y *= r;
  }

  // scalar divide by r
  RAYTRACER_DEVICE_FUNC inline void operator/=(float r) {
    x /= r;
    y /= r;
  }

  /**
   * Returns norm.
   */
  RAYTRACER_DEVICE_FUNC inline const float norm(void) const {
    return sqrt(x * x + y * y);
  }

  /**
   * Returns norm squared.
   */
  RAYTRACER_DEVICE_FUNC inline const float norm2(void) const {
    return x * x + y * y;
  }

  /**
   * Returns unit vector parallel to this one.
   */
  RAYTRACER_DEVICE_FUNC inline Vector2f unit(void) const {
    return *this / this->norm();
  }


}; // clasd Vector2D

// left scalar multiplication
RAYTRACER_DEVICE_FUNC inline const Vector2f operator*(float r, const Vector2f &v) {
  return v * r;
}
