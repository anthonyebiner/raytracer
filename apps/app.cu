#include "fmt/core.h"
#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/renderer.cuh"
#include "geometry/generator.cuh"

using fmt::print;

int main() {
  printf("Starting program\n");

  PathTracer pathtracer = PathTracer({1000, 1, 50, 0.005f, 7, 4});

  std::vector<Primitive *> primitives;
  Generator::room(&primitives, 200, 150, 200,
                  &white_bsdf, &white_bsdf, &red_bsdf, &blue_bsdf, nullptr, &white_bsdf);

  primitives.push_back(new Primitive(PrimitiveFactory::createSphere({-40, 25, 30}, 25, &mirror_bsdf)));
  primitives.push_back(new Primitive(PrimitiveFactory::createSphere({40, 25, -30}, 25, &glass_bsdf)));

  BSDF white_light_bsdf = BSDFFactory::createEmission({7.5, 7.5, 7.5});
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-40, 149.1, -30), Vector3f(40, 149.1, -30), Vector3f(40, 149.1, 30),
                                       Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), &white_light_bsdf)));
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-40, 149.1, -30), Vector3f(-40, 149.1, 30), Vector3f(40, 149.1, 30),
                                       Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), &white_light_bsdf)));

  print("{} primitives\n", primitives.size());

  std::vector<SceneLight> lights;
  lights.push_back(
      SceneLightFactory::create_area({15, 15, 15}, {0, 149, 0}, {0, -1, 0}, {80, 0, 0}, {0, 0, 60}));

  pathtracer.set_scene(primitives, lights);
  pathtracer.resize(1280, 1024);

  pathtracer.set_camera({0, 75, -360}, {0, 75, 0}, {0, 1, 0}, 36.75, 49, 0, INF_F, 0, 0);

  pathtracer.raytrace();
  pathtracer.save_to_file("test1.bmp");
}
