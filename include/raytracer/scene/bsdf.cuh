#pragma once

#include "raytracer/util/sampler.cuh"
#include "Eigen/Dense"

using Eigen::Array3f;
using Eigen::Vector3f;
using Eigen::Matrix3f;

__device__ __host__ void make_coord_space(Matrix3f &o2w, const Vector3f &n) {
  Vector3f z = {n.x(), n.y(), n.z()};
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
    PHONG,
    INVALID,
  } type;

  union {
    struct {
      Array3f reflectance;
    } diffuse;
    struct {
      Array3f radiance;
    } emission;
    struct {
      float roughness;
      Array3f reflectance;
    } reflection;
    struct {
      float ior, roughness;
      Array3f transmittance;
    } refraction;
    struct {
      float ior, roughness;
      Array3f reflectance, transmittance;
    } glass;
    struct {
      float alpha;
      Array3f eta, k;
    } microfacet;
    struct {
      float shininess;
      Array3f diffuse, specular;
    } phong;
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
      case PHONG: {
        phong = b.phong;
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
      case PHONG: {
        phong = b.phong;
      }
      case INVALID:
        break;
    }
    return *this;
  }

  __device__ __host__ bool is_delta() const {
    switch (type) {
      case REFLECTION: {
        return true;
      }
      case REFRACTION: {
        return true;
      }
      case GLASS: {
        return true;
      }
      default: {
        return false;
      }
    }
  }

  __device__ __host__ Vector3f get_emission() const {
    switch (type) {
      case EMISSION: {
        return emission.radiance;
      }
      default: {
        return {0, 0, 0};
      }
    }
  }

  __device__ __host__ Array3f f(const Vector3f &o_out, const Vector3f &o_in) {
    switch (type) {
      case DIFFUSE: {
        return diffuse.reflectance / PI;
      }
      case MICROFACET: {
        if (o_out.z() <= 0 || o_in.z() <= 0) {
          return {0, 0, 0};
        }
        Vector3f h = (o_out + o_in).normalized();
        return (fresnel(o_in, microfacet.eta, microfacet.k) * shadow_masking(o_out, o_in, microfacet.alpha) *
                ndf(h, microfacet.alpha)) / (4 * o_out.z() * o_in.z());
      }
      case PHONG: {
        Vector3f h = (o_out + o_in).normalized();
        return (phong.specular * (phong.shininess + 2) * pow(h.z(), phong.shininess) / 2) / PI;
      }
      default: {
        return {0, 0, 0};
      }
    }
  }

  __device__ __host__ Vector3f
  sample(const Vector3f &o_out, Array3f *mask, float *pdf, uint *seed) {
    float cos_o_out = o_out.z();
    Vector3f o_in;
    switch (type) {
      case DIFFUSE: {
        o_in = Sampler3D::sample_cosine_weighted_hemisphere(seed);
        *pdf = o_in.z() / PI;
        *mask = f(o_out, o_in);
        return o_in;
      }
      case EMISSION: {
        o_in = Sampler3D::sample_cosine_weighted_hemisphere(seed);
        *pdf = 1 / PI;
        *mask = f(o_out, o_in);
        return o_in;
      }
      case REFLECTION: {
        reflect(o_out, &o_in);
        *pdf = o_in.z();
        *mask = reflection.reflectance;
        return o_in;
      }
      case REFRACTION: {
        if (!refract(o_out, &o_in, refraction.ior)) {
          *pdf = 0;
          *mask = {0, 0, 0};
          return o_in;
        }
        float n = cos_o_out > 0 ? 1 / refraction.ior : refraction.ior;
        *pdf = o_in.z();
        *mask = refraction.transmittance / powf(n, 2);
        return o_in;
      }
      case GLASS: {
        if (!refract(o_out, &o_in, glass.ior)) {
          reflect(o_out, &o_in);
          *pdf = o_in.z();
          *mask = glass.reflectance;
          return o_in;
        } else {
          float r0 = pow((1 - glass.ior) / (1 + glass.ior), 2);
          float r = r0 + (1 - r0) * pow(1 - abs(o_out.z()), 5);
          if (Sampler1D::coin_flip(seed, r)) {
            reflect(o_out, &o_in);
            *pdf = r * o_in.z();
            *mask = r * glass.reflectance;
            return o_in;
          } else {
            float n = cos_o_out > 0 ? 1 / glass.ior : glass.ior;
            *pdf = (1 - r) * o_in.z();
            *mask = (1 - r) * glass.transmittance / pow(n, 2);
            return o_in;
          }
        }
      }
      case MICROFACET: {
        Vector2f r = Sampler2D::sample_grid(seed);

        float a2 = pow(microfacet.alpha, 2);
        float t = atan(sqrt(-a2 * log(1 - r.x())));
        float p = 2 * PI * r.y();

        Vector3f h = {sinf(t) * cosf(p), sinf(t) * sinf(p), cosf(t)};
        o_in = -o_out + 2 * o_out.dot(h) * h;

        float pt = 2 * sin(t) / (a2 * pow(cos(t), 3)) * exp(-pow(tan(t), 2) / a2);
        float pp = 1 / (2 * PI);
        float pw = (pt * pp) / sin(t);
        *pdf = pw / (4 * o_in.dot(h));
        *mask = f(o_out, o_in);
        return o_in;
      }
      case PHONG: {
        float cpdf = 0;
        float c = Sampler1D::random(seed);
        if (c < cpdf) {
          o_in = Sampler3D::sample_cosine_weighted_hemisphere(seed);
          *pdf = cpdf * o_in.z() / PI;
          *mask = phong.diffuse / PI;
          return o_in;
        } else {
          Vector2f r = Sampler2D::sample_grid(seed);

          float a = acosf(pow(r[0], 1 / (phong.shininess + 1)));
          float t = a + acos(o_out[2]);
          float p = 2 * PI * r[1] + atan(o_out[1] / o_out[0]);

          *pdf = (1 - cpdf) * (phong.shininess + 2) / (2 * PI) * pow(cos(a), phong.shininess);
          *mask = phong.specular * (phong.shininess + 2) * pow(cos(a), phong.shininess) / (2 * PI);
          return {sinf(t) * cosf(p), sinf(t) * sinf(p), cosf(t)};
        }
      }
      case INVALID: {
        *pdf = 0;
        *mask = {0, 0, 0};
        o_in = {0, 0, 1};
        return o_in;
      }
    }
  }

  static __device__ __host__ float cos_theta(const Vector3f &w) {
    return w.z();
  }

  static __device__ __host__ float acos_theta(const Vector3f &w) {
    return acosf(min(max(w.z(), -1.0f + 1e-5f), 1.0f - 1e-5f));
  }

  static __device__ __host__ float lambda(const Vector3f &w, float alpha) {
    float theta = acos_theta(w);
    float a = 1.0f / (alpha * tanf(theta));
    return 0.5f * (erff(a) - 1.0f + expf(-a * a) / (a * PI));
  }

  static __device__ __host__ double shadow_masking(const Vector3f &o_out, const Vector3f &wi, float alpha) {
    return 1.0 / (1.0 + lambda(wi, alpha) + lambda(o_out, alpha));
  }

  static __device__ __host__ double ndf(const Vector3f &h, float alpha) {
    double t = acos_theta(h);
    double a2 = alpha * alpha;
    return exp(-(pow(tan(t), 2) / a2)) / (PI * a2 * pow(cos(t), 4));
  }

  static __device__ __host__ Array3f fresnel(const Vector3f &o_in, const Array3f &eta, const Array3f &k) {
    float t = acos_theta(o_in);
    Array3f a = eta * eta + k * k;
    Array3f b = 2 * eta * cos(t);
    float c = pow(cos(t), 2);
    Array3f rs = (a - b + c) / (a + b + c);
    Array3f rp = (a * c - b + 1) / (a * c + b + 1);
    return (rs + rp) / 2;
  }

  static __device__ __host__ void reflect(const Vector3f &wo, Vector3f *wi) {
    Vector3f out = {-wo.x(), -wo.y(), wo.z()};
    *wi = out;
  }

  static __device__ __host__ bool refract(const Vector3f &wo, Vector3f *wi, float ior) {
    float n = wo.z() > 0 ? 1 / ior : ior;
    float n2 = powf(n, 2);
    float z2 = powf(wo.z(), 2);

    if (1 - n2 * (1 - z2) < 0) {
      return false;
    }

    Vector3f out = {-n * wo.x(), -n * wo.y(), sqrt(1 - n2 * (1 - z2))};
    if (wo.z() > 0) {
      out.z() = -out.z();
    }
    *wi = out;
    return true;
  }
};


class BSDFFactory {
public:
  __host__ static BSDF createDiffuse(const Array3f &reflectance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::DIFFUSE;
    bsdf.diffuse = {reflectance};
    return bsdf;
  }

  __host__ static BSDF createEmission(const Array3f &radiance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::EMISSION;
    bsdf.emission = {radiance};
    return bsdf;
  }

  __host__ static BSDF createReflection(float roughness, const Array3f &reflectance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::REFLECTION;
    bsdf.reflection = {roughness, reflectance};
    return bsdf;
  }

  __host__ static BSDF createRefraction(float ior, float roughness, const Array3f &transmittance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::REFRACTION;
    bsdf.refraction = {ior, roughness, transmittance};
    return bsdf;
  }

  __host__ static BSDF createGlass(float ior, float roughness,
                                   const Array3f &reflectance, const Array3f &transmittance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::GLASS;
    bsdf.glass = {ior, roughness, reflectance, transmittance};
    return bsdf;
  }

  __host__ static BSDF createMicrofacet(float alpha, const Array3f &eta, const Array3f &k) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::MICROFACET;
    bsdf.microfacet = {alpha, eta, k};
    return bsdf;
  }

  __host__ static BSDF createPhong(float shininess, const Array3f &diffuse, const Array3f &specular) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::PHONG;
    bsdf.phong = {shininess, diffuse, specular};
    return bsdf;
  }
};

static BSDF mirror_bsdf = BSDFFactory::createReflection(0.f, {1, 1, 1});
static BSDF refract_bsdf = BSDFFactory::createRefraction(1.45f, 0.f, {1, 1, 1});
static BSDF glass_bsdf = BSDFFactory::createGlass(1.45f, 0.f, {1, 1, 1}, {1, 1, 1});
static BSDF gold_bsdf = BSDFFactory::createMicrofacet(0.3, {0.21646, 0.42833, 1.3284}, {3.2390, 2.4599, 1.8661});
static BSDF copper_bsdf = BSDFFactory::createMicrofacet(0.05, {0.33228, 1.0162, 1.2474}, {3.1646, 2.5785, 2.4603});
static BSDF iron_bsdf = BSDFFactory::createMicrofacet(0.3, {2.8851, 2.9500, 2.6500}, {3.2419, 2.9300, 2.8075});

static BSDF black_bsdf = BSDFFactory::createDiffuse({0, 0, 0});
static BSDF white_bsdf = BSDFFactory::createDiffuse({.6, .6, .6});
static BSDF red_bsdf = BSDFFactory::createDiffuse({.6, .2, .2});
static BSDF green_bsdf = BSDFFactory::createDiffuse({.2, .6, .2});
static BSDF blue_bsdf = BSDFFactory::createDiffuse({.2, .2, .6});
