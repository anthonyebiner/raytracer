#include "fmt/core.h"
#include "physics/physics.cuh"
#include "raytracer/renderer.cuh"
#include "geometry/generator.cuh"

using fmt::print;

int main() {
  printf("Starting program\n");

  std::vector<SceneLight> lights;
  lights.push_back(SceneLightFactory::create_directional({.2, .2, .2}, {0, -1, 0}));


  std::vector<Primitive *> primitives;
  PhysicsSystem physics = PhysicsSystem(500000, {0, 0, 0}, {0, 1, 0}, 250);

  for (uint i = 0; i < 80; i++) {
    physics.step(.075);
    physics.to_primitives(&primitives);
    Generator::room(&primitives, 10000, 0, 10000, &white_bsdf, nullptr, nullptr,
                    nullptr, nullptr, nullptr);

    PathTracer pathtracer = PathTracer({100, 1, 64, 0.05f, 8, 4});
    pathtracer.set_camera({0, 175, -300}, {0, 20, 0}, {0, 1, 0}, 42, 42 * 600 / 800, 0, INF_F, 0, 0);
    pathtracer.resize(800, 600);
    pathtracer.set_scene(primitives, lights);

    pathtracer.raytrace();
    pathtracer.save_to_file("sim" + std::to_string(i) + ".bmp");

    for (const auto &item: primitives) {
      if (item->type != Primitive::TRIANGLE)
        delete item->bsdf;
    }
    primitives.clear();
  }
}
