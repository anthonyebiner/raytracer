#pragma once

#include "raytracer/util/sampler.cuh"
#include "Eigen/Dense"

using Eigen::Vector3f;
using Eigen::Matrix3f;

__device__ __host__ void make_coord_space(Matrix3f &o2w, const Vector3f &n) {
  Vector3f z = Vector3f(n.x(), n.y(), n.z());
  Vector3f h = z;
  if (fabs(h.x()) <= fabs(h.y()) && fabs(h.x()) <= fabs(h.z()))
    h.x() = 1.0;
  else if (fabs(h.y()) <= fabs(h.x()) && fabs(h.y()) <= fabs(h.z()))
    h.y() = 1.0;
  else
    h.z() = 1.0;

  z.normalize();
  Vector3f y = h.cross(z);
  y.normalize();
  Vector3f x = z.cross(y);
  x.normalize();

  o2w.col(0) = x;
  o2w.col(1) = y;
  o2w.col(2) = z;
}


class BSDF {
public:
  enum {
    DIFFUSE,
    EMISSION,
    REFLECTION,
    REFRACTION,
    GLASS,
    MICROFACET,
    INVALID,
  } type;

  union {
    struct {
      Vector3f reflectance;
    } diffuse;
    struct {
      Vector3f radiance;
    } emission;
    struct {
      float roughness;
      Vector3f reflectance;
    } reflection;
    struct {
      float ior, roughness;
      Vector3f transmittance;
    } refraction;
    struct {
      float ior, roughness;
      Vector3f reflectance, transmittance;
    } glass;
    struct {
      float alpha;
      Vector3f eta, k;
    } microfacet;
  };

  __host__ explicit BSDF() : type(INVALID) {}

  __host__ BSDF(BSDF const &b) {
    type = b.type;
    switch (type) {
      case DIFFUSE: {
        diffuse = b.diffuse;
      }
      case EMISSION: {
        emission = b.emission;
      }
      case REFLECTION: {
        reflection = b.reflection;
      }
      case REFRACTION: {
        refraction = b.refraction;
      }
      case GLASS: {
        glass = b.glass;
      }
      case MICROFACET: {
        microfacet = b.microfacet;
      }
      case INVALID:
        break;
    }
  }

  BSDF &operator=(BSDF const &b) {
    type = b.type;
    switch (type) {
      case DIFFUSE: {
        diffuse = b.diffuse;
      }
      case EMISSION: {
        emission = b.emission;
      }
      case REFLECTION: {
        reflection = b.reflection;
      }
      case REFRACTION: {
        refraction = b.refraction;
      }
      case GLASS: {
        glass = b.glass;
      }
      case MICROFACET: {
        microfacet = b.microfacet;
      }
      case INVALID:
        break;
    }
    return *this;
  }

  __device__ bool is_lit() const {
    switch (type) {
      case REFLECTION: {
        return reflection.roughness != 0;
      }
      case REFRACTION: {
        return refraction.roughness != 0;
      }
      case GLASS: {
        return glass.roughness != 0;
      }
      default: {
        return true;
      }
    }
  }

  __device__ Vector3f get_emission() const {
    switch (type) {
      case EMISSION: {
        return emission.radiance;
      }
      default: {
        return Vector3f::Zero();
      }
    }
  }

  __device__ Vector3f f(const Vector3f &wo, const Vector3f &wi) {
    switch (type) {
      case DIFFUSE: {
        return diffuse.reflectance / PI;
      }
      case REFLECTION: {
        return reflection.reflectance / PI * reflection.roughness;
      }
      case REFRACTION: {
        return refraction.transmittance / PI * refraction.roughness;
      }
      case GLASS: {
        return glass.reflectance / PI * glass.roughness;
      }
      case MICROFACET: {
        if (wo.z() <= 0 || wi.z() <= 0) {
          return Vector3f::Zero();
        }
        Vector3f h = (wo + wi).normalized();
        return (fresnel(wi, microfacet.eta, microfacet.k) * shadow_masking(wo, wi, microfacet.alpha) *
                ndf(h, microfacet.alpha)) / (4 * wo.z() * wi.z());
      }
      default: {
        return Vector3f::Zero();
      }
    }
  }

  __device__ Vector3f sample(const Vector3f &o_out, Vector3f *o_in, float *pdf, curandState *rand_state) {
    switch (type) {
      case DIFFUSE: {
        *o_in = CudaSampler3D::sample_cosine_weighted_hemisphere(rand_state);
        *pdf = 1 / PI;
        return f(o_out, *o_in) / cos_theta(*o_in);
      }
      case EMISSION: {
        *o_in = CudaSampler3D::sample_cosine_weighted_hemisphere(rand_state);
        *pdf = 1;
        return emission.radiance / cos_theta(*o_in);
      }
      case REFLECTION: {
        reflect(o_out, o_in);
        *pdf = 1;
        return reflection.reflectance / cos_theta(*o_in);
      }
      case REFRACTION: {
        *pdf = 1;
        if (!refract(o_out, o_in, refraction.ior)) {
          return Vector3f::Zero();
        }
        float n = o_out.z() > 0 ? 1 / refraction.ior : refraction.ior;
        return refraction.transmittance / cos_theta(*o_in) / powf(n, 2);
      }
      case GLASS: {
        if (!refract(o_out, o_in, glass.ior)) {
          reflect(o_out, o_in);
          *pdf = 1;
          return glass.reflectance / cos_theta(*o_in);
        } else {
          float r0 = pow((1 - glass.ior) / (1 + glass.ior), 2);
          float r = r0 + (1 - r0) * pow(1 - abs(o_out.z()), 5);
          if (CudaSampler1D::coin_flip(rand_state, r)) {
            reflect(o_out, o_in);
            *pdf = r;
            return r * glass.reflectance / cos_theta(*o_in);
          } else {
            float n = o_out.z() > 0 ? 1 / glass.ior : glass.ior;
            *pdf = 1 - r;
            return (1 - r) * glass.transmittance / cos_theta(*o_in) / pow(n, 2);
          }
        }
      }
      case MICROFACET: {
        Vector2f r = CudaSampler2D::sample_grid(rand_state);

        float a2 = pow(microfacet.alpha, 2);
        float t = atan(sqrt(-a2 * log(1 - r.x())));
        float p = 2 * PI * r.y();

        auto h = Vector3f(sinf(t) * cosf(p), sinf(t) * sinf(p), cosf(t));
        *o_in = -o_out + 2 * o_out.dot(h) * h;
        if (o_in->z() <= 0) {
//          *pdf = 1;
          return Vector3f::Zero();
        }

        float pt = 2 * sin(t) / (a2 * pow(cos(t), 3)) * exp(-pow(tan(t), 2) / a2);
        float pp = 1 / (2 * PI);
        float pw = (pt * pp) / sin(t);
//        *pdf = pw / (4 * (*o_in).dot(h));
        return f(o_out, *o_in);
      }
      case INVALID: {
      }
    }
  }

  static __device__ float cos_theta(const Vector3f &w) {
    return w.z();
  }

  static __device__ float acos_theta(const Vector3f &w) {
    return acosf(min(max(w.z(), -1.0f + 1e-5f), 1.0f - 1e-5f));
  }

  static __device__ float lambda(const Vector3f &w, float alpha) {
    float theta = acos_theta(w);
    float a = 1.0f / (alpha * tanf(theta));
    return 0.5f * (erff(a) - 1.0f + expf(-a * a) / (a * PI));
  }

  static __device__ double shadow_masking(const Vector3f &wo, const Vector3f &wi, float alpha) {
    return 1.0 / (1.0 + lambda(wi, alpha) + lambda(wo, alpha));
  }

  static __device__ double ndf(const Vector3f &h, float alpha) {
    double t = acos_theta(h);
    double a2 = alpha * alpha;
    return exp(-(pow(tan(t), 2) / a2)) / (PI * a2 * pow(cos(t), 4));
  }

  static __device__ Vector3f fresnel(const Vector3f &wi, const Vector3f &eta, const Vector3f &k) {
    float t = acos_theta(wi);
    Vector3f a = Vector3f(eta.array() * eta.array() + k.array() * k.array());
    Vector3f b = 2 * eta * cos(t);
    float c = pow(cos(t), 2);
    Vector3f rs = Vector3f((a.array() - b.array() + c) / (a.array() + b.array() + c));
    Vector3f rp = Vector3f((a.array() * c - b.array() + 1) / (a.array() * c + b.array() + 1));
    return (rs + rp) / 2;
  }

  static __device__ void reflect(const Vector3f &wo, Vector3f *wi) {
    Vector3f out = Vector3f(-wo.x(), -wo.y(), wo.z());
    *wi = out;
  }

  static __device__ bool refract(const Vector3f &wo, Vector3f *wi, float ior) {
    float n = wo.z() > 0 ? 1 / ior : ior;
    float n2 = powf(n, 2);
    float z2 = powf(wo.z(), 2);

    if (1 - n2 * (1 - z2) < 0) {
      return false;
    }

    Vector3f out = Vector3f(-n * wo.x(), -n * wo.y(), sqrt(1 - n2 * (1 - z2)));
    if (wo.z() > 0) {
      out.z() = -out.z();
    }
    *wi = out;
    return true;
  }
};


class BSDFFactory {
public:
  __host__ static BSDF createDiffuse(const Vector3f &reflectance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::DIFFUSE;
    bsdf.diffuse = {reflectance};
    return bsdf;
  }

  __host__ static BSDF createEmission(const Vector3f &radiance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::EMISSION;
    bsdf.emission = {radiance};
    return bsdf;
  }

  __host__ static BSDF createReflection(float roughness, const Vector3f &reflectance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::REFLECTION;
    bsdf.reflection = {roughness, reflectance};
    return bsdf;
  }

  __host__ static BSDF createRefraction(float ior, float roughness, const Vector3f &transmittance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::REFRACTION;
    bsdf.refraction = {ior, roughness, transmittance};
    return bsdf;
  }

  __host__ static BSDF createGlass(float ior, float roughness,
                                   const Vector3f &reflectance, const Vector3f &transmittance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::GLASS;
    bsdf.glass = {ior, roughness, reflectance, transmittance};
    return bsdf;
  }

  __host__ static BSDF createMicrofacet(float alpha, const Vector3f &eta, const Vector3f &k) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::MICROFACET;
    bsdf.microfacet = {alpha, eta, k};
    return bsdf;
  }
};
