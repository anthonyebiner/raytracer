#pragma once

#include "raytracer/linalg/Matrix3f.cuh"
#include "raytracer/linalg/Vector2f.cuh"
#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/util/sampler.cuh"


RAYTRACER_DEVICE_FUNC void make_coord_space(Matrix3f &o2w, const Vector3f &n) {
  Vector3f z = {n.x, n.y, n.z};
  Vector3f h = z;
  if (fabsf(h.x) <= fabsf(h.y) && fabsf(h.x) <= fabsf(h.z))
    h.x = 1.0;
  else if (fabsf(h.y) <= fabsf(h.x) && fabsf(h.y) <= fabsf(h.z))
    h.y = 1.0;
  else
    h.z = 1.0;

  z.normalize();
  Vector3f y = h.cross(z);
  y.normalize();
  Vector3f x = z.cross(y);
  x.normalize();

  o2w[0] = x;
  o2w[1] = y;
  o2w[2] = z;
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

  explicit BSDF() : type(INVALID) {}

  BSDF(BSDF const &b) {
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

  RAYTRACER_DEVICE_FUNC bool is_delta() const {
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

  RAYTRACER_DEVICE_FUNC Vector3f get_emission() const {
    switch (type) {
      case EMISSION: {
        return emission.radiance;
      }
      default: {
        return {0, 0, 0};
      }
    }
  }

  RAYTRACER_DEVICE_FUNC Vector3f f(const Vector3f &o_out, const Vector3f &o_in) {
    switch (type) {
      case DIFFUSE: {
        return diffuse.reflectance / PI;
      }
      case MICROFACET: {
        if (o_out.z <= 0 || o_in.z <= 0) {
          return {0, 0, 0};
        }
        Vector3f h = (o_out + o_in).unit();
        return (fresnel(o_in, microfacet.eta, microfacet.k) * shadow_masking(o_out, o_in, microfacet.alpha) *
                ndf(h, microfacet.alpha)) / (4 * o_out.z * o_in.z);
      }
      default: {
        return {0, 0, 0};
      }
    }
  }

  RAYTRACER_DEVICE_FUNC Vector3f
  sample(Vector3f o_out, Vector3f *mask, float *pdf, uint *seed) {
    float cos_o_out = o_out.z;
    Vector3f o_in;
    switch (type) {
      case DIFFUSE: {
        o_in = Sampler3D::sample_cosine_weighted_hemisphere(seed);
        *pdf = o_in.z / PI;
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
        *pdf = o_in.z;
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
        *pdf = o_in.z;
        *mask = refraction.transmittance / powf(n, 2);
        return o_in;
      }
      case GLASS: {
        if (!refract(o_out, &o_in, glass.ior)) {
          reflect(o_out, &o_in);
          *pdf = o_in.z;
          *mask = glass.reflectance;
          return o_in;
        } else {
          float r0 = pow((1 - glass.ior) / (1 + glass.ior), 2);
          float r = r0 + (1 - r0) * pow(1 - abs(o_out.z), 5);
          if (Sampler1D::coin_flip(seed, r)) {
            reflect(o_out, &o_in);
            *pdf = r * o_in.z;
            *mask = r * glass.reflectance;
            return o_in;
          } else {
            float n = cos_o_out > 0 ? 1 / glass.ior : glass.ior;
            *pdf = (1 - r) * o_in.z;
            *mask = (1 - r) * glass.transmittance / pow(n, 2);
            return o_in;
          }
        }
      }
      case MICROFACET: {
        Vector2f r = Sampler2D::sample_grid(seed);

        float a2 = pow(microfacet.alpha, 2);
        float t = atan(sqrt(-a2 * log(1 - r.x)));
        float p = 2 * PI * r.y;

        Vector3f h = {sinf(t) * cosf(p), sinf(t) * sinf(p), cosf(t)};
        o_in = -o_out + 2 * o_out.dot(h) * h;

        float pt = 2 * sin(t) / (a2 * pow(cos(t), 3)) * exp(-pow(tan(t), 2) / a2);
        float pp = 1 / (2 * PI);
        float pw = (pt * pp) / sin(t);
        *pdf = pw / (4 * o_in.dot(h));
        *mask = f(o_out, o_in);
        return o_in;
      }
      case INVALID: {
        *pdf = 0;
        *mask = {0, 0, 0};
        o_in = {0, 0, 1};
        return o_in;
      }
    }
  }

  static RAYTRACER_DEVICE_FUNC float cos_theta(const Vector3f &w) {
    return w.z;
  }

  static RAYTRACER_DEVICE_FUNC float acos_theta(const Vector3f &w) {
    return acosf(minf(maxf(w.z, -1.0f + 1e-5f), 1.0f - 1e-5f));
  }

  static RAYTRACER_DEVICE_FUNC float lambda(const Vector3f &w, float alpha) {
    float theta = acos_theta(w);
    float a = 1.0f / (alpha * tanf(theta));
    return 0.5f * (erff(a) - 1.0f + expf(-a * a) / (a * PI));
  }

  static RAYTRACER_DEVICE_FUNC double shadow_masking(const Vector3f &o_out, const Vector3f &wi, float alpha) {
    return 1.0 / (1.0 + lambda(wi, alpha) + lambda(o_out, alpha));
  }

  static RAYTRACER_DEVICE_FUNC double ndf(const Vector3f &h, float alpha) {
    double t = acos_theta(h);
    double a2 = alpha * alpha;
    return exp(-(pow(tan(t), 2) / a2)) / (PI * a2 * pow(cos(t), 4));
  }

  static RAYTRACER_DEVICE_FUNC Vector3f fresnel(const Vector3f &o_in, const Vector3f &eta, const Vector3f &k) {
    float t = acos_theta(o_in);
    Vector3f a = eta * eta + k * k;
    Vector3f b = 2 * eta * cos(t);
    float c = pow(cos(t), 2);
    Vector3f rs = (a - b + c) / (a + b + c);
    Vector3f rp = (a * c - b + 1) / (a * c + b + 1);
    return (rs + rp) / 2;
  }

  static RAYTRACER_DEVICE_FUNC void reflect(const Vector3f &wo, Vector3f *wi) {
    Vector3f out = {-wo.x, -wo.y, wo.z};
    *wi = out;
  }

  static RAYTRACER_DEVICE_FUNC bool refract(const Vector3f &wo, Vector3f *wi, float ior) {
    float n = wo.z > 0 ? 1 / ior : ior;
    float n2 = powf(n, 2);
    float z2 = powf(wo.z, 2);

    if (1 - n2 * (1 - z2) < 0) {
      return false;
    }

    Vector3f out = {-n * wo.x, -n * wo.y, sqrtf(1 - n2 * (1 - z2))};
    if (wo.z > 0) {
      out.z = -out.z;
    }
    *wi = out;
    return true;
  }
};


class BSDFFactory {
public:
  static BSDF createDiffuse(const Vector3f &reflectance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::DIFFUSE;
    bsdf.diffuse = {reflectance};
    return bsdf;
  }

  static BSDF createEmission(const Vector3f &radiance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::EMISSION;
    bsdf.emission = {radiance};
    return bsdf;
  }

  static BSDF createReflection(float roughness, const Vector3f &reflectance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::REFLECTION;
    bsdf.reflection = {roughness, reflectance};
    return bsdf;
  }

  static BSDF createRefraction(float ior, float roughness, const Vector3f &transmittance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::REFRACTION;
    bsdf.refraction = {ior, roughness, transmittance};
    return bsdf;
  }

  static BSDF createGlass(float ior, float roughness,
                          const Vector3f &reflectance, const Vector3f &transmittance) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::GLASS;
    bsdf.glass = {ior, roughness, reflectance, transmittance};
    return bsdf;
  }

  static BSDF createMicrofacet(float alpha, const Vector3f &eta, const Vector3f &k) {
    auto bsdf = BSDF();
    bsdf.type = BSDF::MICROFACET;
    bsdf.microfacet = {alpha, eta, k};
    return bsdf;
  }

};

static BSDF mirror_bsdf = BSDFFactory::createReflection(0.f, {.9, .9, .9});
static BSDF refract_bsdf = BSDFFactory::createRefraction(1.45f, 0.f, {1, 1, 1});
static BSDF glass_bsdf = BSDFFactory::createGlass(1.45f, 0.f, {1, 1, 1}, {1, 1, 1});

static BSDF aluminum_bsdf = BSDFFactory::createMicrofacet(0.5, {1.34560, 0.96521, 0.61722},
                                                          {7.47460, 6.39950, 5.30310});
static BSDF brass_bsdf = BSDFFactory::createMicrofacet(0.5, {0.44400, 0.52700, 1.09400},
                                                       {3.69500, 2.76500, 1.82900});
static BSDF copper_bsdf = BSDFFactory::createMicrofacet(0.5, {0.27105, 0.67693, 1.31640},
                                                        {3.60920, 2.62480, 2.29210});
static BSDF gold_bsdf = BSDFFactory::createMicrofacet(0.05, {0.18299, 0.42108, 1.37340},
                                                      {3.42420, 2.34590, 1.77040});
static BSDF iron_bsdf = BSDFFactory::createMicrofacet(0.5, {2.91140, 2.94970, 2.58450},
                                                      {3.08930, 2.93180, 2.76700});
static BSDF lead_bsdf = BSDFFactory::createMicrofacet(0.5, {1.91000, 1.83000, 1.44000},
                                                      {3.51000, 3.40000, 3.18000});
static BSDF mercury_bsdf = BSDFFactory::createMicrofacet(0.5, {2.07330, 1.55230, 1.06060},
                                                         {5.33830, 4.65100, 3.86280});
static BSDF platinum_bsdf = BSDFFactory::createMicrofacet(0.5, {2.37570, 2.08470, 1.84530},
                                                          {4.26550, 3.71530, 3.13650});
static BSDF silver_bsdf = BSDFFactory::createMicrofacet(0.5, {0.15943, 0.14512, 0.13547},
                                                        {3.92910, 3.19000, 2.38080});
static BSDF titanium_bsdf = BSDFFactory::createMicrofacet(0.5, {2.74070, 2.54180, 2.26700},
                                                          {3.81430, 3.43450, 3.03850});
static BSDF blue_metal_bsdf = BSDFFactory::createMicrofacet(0.5, {2.5, .2, .05}, {0.5, 0.5, 0.5});

static BSDF black_bsdf = BSDFFactory::createDiffuse({0, 0, 0});
static BSDF white_bsdf = BSDFFactory::createDiffuse({.6, .6, .6});
static BSDF red_bsdf = BSDFFactory::createDiffuse({.6, .2, .2});
static BSDF green_bsdf = BSDFFactory::createDiffuse({.2, .6, .2});
static BSDF blue_bsdf = BSDFFactory::createDiffuse({.2, .2, .6});
