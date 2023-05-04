#include "fmt/core.h"
#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/renderer.cuh"
#include "geometry/generator.cuh"

using fmt::print;

int main() {
  printf("Starting program\n");

  PathTracer pathtracer = PathTracer({100, 4, 64, 0.05f, 40, 4});

  std::vector<Primitive *> primitives;
  Generator::room(&primitives, 2, 1.5, 2,
                  &white_bsdf, &white_bsdf, &red_bsdf, &blue_bsdf, nullptr, &white_bsdf);

//  Generator::from_obj(&primitives, "../objects/hairball.obj", &white_bsdf, {0.12, 0.12, 0.12}, {0, .5, 0});

  BSDF white_light_bsdf = BSDFFactory::createEmission({8, 8, 8});
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-.3, 1.49, -.3), Vector3f(.3, 1.49, -.3), Vector3f(.3, 1.49, .3),
                                       Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), &white_light_bsdf)));
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-.3, 1.49, -.3), Vector3f(-.3, 1.49, .3), Vector3f(.3, 1.49, .3),
                                       Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), &white_light_bsdf)));

  print("{} primitives\n", primitives.size());

  std::vector<SceneLight> lights;
  lights.push_back(SceneLightFactory::create_area({16, 16, 16}, {0, 1.49, 0}, {0, -1, 0}, {.6, 0, 0}, {0, 0, .6}));

  pathtracer.set_scene(primitives, lights);
  pathtracer.resize(1920, 1080);

  pathtracer.set_camera({-.75, 1.1, -2.75}, {0, .5, 0}, {0, 1, 0}, 42, 42 * 1024 / 1980, 0, INF_F, 0, 0);

  pathtracer.raytrace();
  pathtracer.save_to_file("test3.bmp");
}
