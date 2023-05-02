#pragma once

#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/util/misc.cuh"

/**
 * Defines a 3x3 matrix.
 * 3x3 matrices are extremely useful in computer graphics.
 */
class Matrix3f {
protected:
  Vector3f entries[3];

public:
  // The default constructor. Returns identity.
  RAYTRACER_DEVICE_FUNC Matrix3f(void) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        (*this)(i, j) = (i == j) ? 1. : 0.;
      }
    }
  }

  // Constructor for row major form data.
  // Transposes to the internal column major form.
  // REQUIRES: data should be of size 9 for a 3 by 3 matrix..
  RAYTRACER_DEVICE_FUNC Matrix3f(float *data) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        // Transpostion happens within the () query.
        (*this)(i, j) = data[i * 3 + j];
      }
    }
  }

  RAYTRACER_DEVICE_FUNC Matrix3f(float m00, float m01, float m02,
                                 float m10, float m11, float m12,
                                 float m20, float m21, float m22) {
    (*this)(0, 0) = m00;
    (*this)(0, 1) = m01;
    (*this)(0, 2) = m02;
    (*this)(1, 0) = m10;
    (*this)(1, 1) = m11;
    (*this)(1, 2) = m12;
    (*this)(2, 0) = m20;
    (*this)(2, 1) = m21;
    (*this)(2, 2) = m22;
  }

  /**
   * Sets all elements to val.
   */
  RAYTRACER_DEVICE_FUNC void zero(float val = 0.0) {
    entries[0] = entries[1] = entries[2] = Vector3f(val, val, val);
  }

  /**
   * Returns the determinant of A.
   */
  RAYTRACER_DEVICE_FUNC float det(void) const {
    const Matrix3f &A(*this);

    return -A(0, 2) * A(1, 1) * A(2, 0) + A(0, 1) * A(1, 2) * A(2, 0) +
           A(0, 2) * A(1, 0) * A(2, 1) - A(0, 0) * A(1, 2) * A(2, 1) -
           A(0, 1) * A(1, 0) * A(2, 2) + A(0, 0) * A(1, 1) * A(2, 2);
  }

  /**
   * Returns the Frobenius norm of A.
   */
  RAYTRACER_DEVICE_FUNC float norm(void) const {
    return sqrt(entries[0].norm2() +
                entries[1].norm2() +
                entries[2].norm2());
  }

  /**
   * Returns the 3x3 identity matrix.
   */
  RAYTRACER_DEVICE_FUNC static Matrix3f identity(void) {
    Matrix3f B;

    B(0, 0) = 1.;
    B(0, 1) = 0.;
    B(0, 2) = 0.;
    B(1, 0) = 0.;
    B(1, 1) = 1.;
    B(1, 2) = 0.;
    B(2, 0) = 0.;
    B(2, 1) = 0.;
    B(2, 2) = 1.;

    return B;
  }


  /**
   * Returns a matrix representing the (left) cross product with u.
   */
  RAYTRACER_DEVICE_FUNC static Matrix3f crossProduct(const Vector3f &u) {
    Matrix3f B;

    B(0, 0) = 0.;
    B(0, 1) = -u.z;
    B(0, 2) = u.y;
    B(1, 0) = u.z;
    B(1, 1) = 0.;
    B(1, 2) = -u.x;
    B(2, 0) = -u.y;
    B(2, 1) = u.x;
    B(2, 2) = 0.;

    return B;
  }

  /**
   * Returns the ith column.
   */
  RAYTRACER_DEVICE_FUNC Vector3f &column(int i) {
    return entries[i];
  }

  RAYTRACER_DEVICE_FUNC const Vector3f &column(int i) const {
    return entries[i];
  }

  /**
   * Returns the transpose of A.
   */
  RAYTRACER_DEVICE_FUNC Matrix3f T(void) const {
    const Matrix3f &A(*this);
    Matrix3f B;

    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        B(i, j) = A(j, i);
      }

    return B;
  }

  /**
   * Returns the inverse of A.
   */
  RAYTRACER_DEVICE_FUNC Matrix3f inv(void) const {
    const Matrix3f &A(*this);
    Matrix3f B;

    B(0, 0) = -A(1, 2) * A(2, 1) + A(1, 1) * A(2, 2);
    B(0, 1) = A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2);
    B(0, 2) = -A(0, 2) * A(1, 1) + A(0, 1) * A(1, 2);
    B(1, 0) = A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2);
    B(1, 1) = -A(0, 2) * A(2, 0) + A(0, 0) * A(2, 2);
    B(1, 2) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
    B(2, 0) = -A(1, 1) * A(2, 0) + A(1, 0) * A(2, 1);
    B(2, 1) = A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1);
    B(2, 2) = -A(0, 1) * A(1, 0) + A(0, 0) * A(1, 1);

    B /= det();

    return B;
  }

  // accesses element (i,j) of A using 0-based indexing
  RAYTRACER_DEVICE_FUNC float &operator()(int i, int j) {
    return entries[j][i];
  }

  RAYTRACER_DEVICE_FUNC const float &operator()(int i, int j) const {
    return entries[j][i];
  }


  // accesses the ith column of A
  RAYTRACER_DEVICE_FUNC Vector3f &operator[](int i) {
    return entries[i];
  }

  RAYTRACER_DEVICE_FUNC const Vector3f &operator[](int i) const {
    return entries[i];
  }

  // increments by B
  RAYTRACER_DEVICE_FUNC void operator+=(const Matrix3f &B) {

    Matrix3f &A(*this);

    A[0] += B[0];
    A[1] += B[1];
    A[2] += B[2];
  }

  // returns -A
  Matrix3f operator-(void) const {

    // returns -A
    const Matrix3f &A(*this);
    Matrix3f B;

    B[0] = -A[0];
    B[1] = -A[1];
    B[2] = -A[2];

    return B;
  }

  // returns A-B
  RAYTRACER_DEVICE_FUNC Matrix3f operator-(const Matrix3f &B) const {
    const Matrix3f &A(*this);
    Matrix3f C;

    C[0] = A[0] - B[0];
    C[1] = A[1] - B[1];
    C[2] = A[2] - B[2];

    return C;
  }

  // returns c*A
  RAYTRACER_DEVICE_FUNC Matrix3f operator*(float c) const {
    const Matrix3f &A(*this);
    Matrix3f B;

    B[0] = A[0] * c;
    B[1] = A[1] * c;
    B[2] = A[2] * c;

    return B;
  }

  // returns A*B
  RAYTRACER_DEVICE_FUNC Matrix3f operator*(const Matrix3f &B) const {
    const Matrix3f &A(*this);
    Matrix3f C;

    C[0] = A * B[0];
    C[1] = A * B[1];
    C[2] = A * B[2];

    return C;
  }

  // returns A*x
  RAYTRACER_DEVICE_FUNC Vector3f operator*(const Vector3f &x) const {
    return x.x * entries[0] +
           x.y * entries[1] +
           x.z * entries[2];
  }

  // divides each element by x
  void operator/=(float x) {
    Matrix3f &A(*this);
    double rx = 1. / x;

    A[0] *= rx;
    A[1] *= rx;
    A[2] *= rx;
  }
}; // class Matrix3x3

// returns the outer product of u and v
RAYTRACER_DEVICE_FUNC Matrix3f outer(const Vector3f &u, const Vector3f &v) {
  Matrix3f B;
  double *Bij = (double *) &B;

  *Bij++ = u.x * v.x;
  *Bij++ = u.y * v.x;
  *Bij++ = u.z * v.x;
  *Bij++ = u.x * v.y;
  *Bij++ = u.y * v.y;
  *Bij++ = u.z * v.y;
  *Bij++ = u.x * v.z;
  *Bij++ = u.y * v.z;
  *Bij++ = u.z * v.z;

  return B;
}

// returns c*A
RAYTRACER_DEVICE_FUNC Matrix3f operator*(float c, const Matrix3f &A) {
  Matrix3f cA;

  cA[0] = A[0] * c;
  cA[1] = A[1] * c;
  cA[2] = A[2] * c;

  return cA;
}
