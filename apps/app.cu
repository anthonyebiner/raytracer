#include "fmt/core.h"
#include "raytracer/pathtracer.cuh"
#include <fmt/core.h>

using fmt::print;

int main() {
  printf("Starting program\n");
  PathTracer pathtracer = PathTracer();

  BSDF emissionBSDF = BSDFFactory::createEmission(Vector3f::Ones());
  BSDF reflectionBSDF = BSDFFactory::createReflection(0.05f, Vector3f::Ones());
  BSDF glassBSDF = BSDFFactory::createGlass(1.4f, 0.05f, Vector3f::Ones(), Vector3f::Ones());

  BSDF whiteBSDF = BSDFFactory::createDiffuse({1, 1, 1});
  BSDF redBSDF = BSDFFactory::createDiffuse({.6, .2, .2});
  BSDF greenBSDF = BSDFFactory::createDiffuse({.2, .6, .2});

  std::vector<Primitive *> primitives;
  for (uint i = 0; i < 50; i++) {
    primitives.push_back(new Primitive(PrimitiveFactory::createSphere(Vector3f::Random() * 40, 2.5, &whiteBSDF)));
  }

  for (uint i = 0; i < 15; i++) {
    primitives.push_back(new Primitive(PrimitiveFactory::createSphere(Vector3f::Random() * 40, 3, &reflectionBSDF)));
  }

  for (uint i = 0; i < 7; i++) {
    primitives.push_back(new Primitive(PrimitiveFactory::createSphere(Vector3f::Random() * 40, 5, &glassBSDF)));
  }

  // Floor
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-50, -50, -50), Vector3f(50, -50, -50), Vector3f(50, -50, 50),
                                       Vector3f(0, 1, 0), Vector3f(0, 1, 0), Vector3f(0, 1, 0), &whiteBSDF)));
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-50, -50, -50), Vector3f(-50, -50, 50), Vector3f(50, -50, 50),
                                       Vector3f(0, 1, 0), Vector3f(0, 1, 0), Vector3f(0, 1, 0), &whiteBSDF)));

  // Ceiling
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-50, 50, -50), Vector3f(50, 50, -50), Vector3f(50, 50, 50),
                                       Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), &whiteBSDF)));
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-50, 50, -50), Vector3f(-50, 50, 50), Vector3f(50, 50, 50),
                                       Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), &whiteBSDF)));

  // Left wall
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-50, -50, -50), Vector3f(-50, 50, -50), Vector3f(-50, 50, 50),
                                       Vector3f(1, 0, 0), Vector3f(1, 0, 0), Vector3f(1, 0, 0), &redBSDF)));
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-50, -50, -50), Vector3f(-50, -50, 50), Vector3f(-50, 50, 50),
                                       Vector3f(1, 0, 0), Vector3f(1, 0, 0), Vector3f(1, 0, 0), &redBSDF)));

  // Right wall
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(50, -50, -50), Vector3f(50, 50, -50), Vector3f(50, 50, 50),
                                       Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), &greenBSDF)));
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(50, -50, -50), Vector3f(50, -50, 50), Vector3f(50, 50, 50),
                                       Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), &greenBSDF)));

  // Back wall
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(50, -50, 50), Vector3f(-50, 50, 50), Vector3f(50, 50, 50),
                                       Vector3f(0, 0, -1), Vector3f(0, 0, -1), Vector3f(0, 0, -1), &whiteBSDF)));
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(50, -50, 50), Vector3f(-50, 50, 50), Vector3f(-50, -50, 50),
                                       Vector3f(0, 0, -1), Vector3f(0, 0, -1), Vector3f(0, 0, -1), &whiteBSDF)));

  // Light
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-10, 49, -10), Vector3f(10, 49, -10), Vector3f(10, 49, 10),
                                       Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), &emissionBSDF)));
  primitives.push_back(new Primitive(
      PrimitiveFactory::createTriangle(Vector3f(-10, 49, -10), Vector3f(-10, 49, 10), Vector3f(10, 49, 10),
                                       Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), &emissionBSDF)));

  std::vector<SceneLight> lights;
  lights.push_back(
      SceneLightFactor::create_area(Vector3f(.4, .4, .4), {0, 48.9, 0}, {0, -1, 0}, {20, 0, 0}, {0, 0, 20}));

  pathtracer.set_scene(primitives, lights);
  pathtracer.resize(1280, 1024);
  pathtracer.set_camera({0, 0, -200}, {0, 0, 0}, {0, 1, 0}, 43.52, 54.4, EPS_F, INF_F);

  pathtracer.raytrace();
  pathtracer.save_to_file("test.bmp");
}
