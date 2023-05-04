#pragma once

#include <vector>
#include "raytracer/linalg/Vector3f.cuh"
#include "physics/particle.cuh"
#include "raytracer/scene/primitives.cuh"
#include "raytracer/scene/bsdf.cuh"


class PhysicsSystem {
public:
  std::vector<Particle *> particles;

  PhysicsSystem(uint num_particles, Vector3f starting_position, Vector3f up, float max_starting_velocity) {
    Matrix3f o2w;
    make_coord_space(o2w, up);
    uint seed = 0x12345678;
    std::vector<Vector3f> splotches;
    for (uint i = 0; i < 20; i++) {
      splotches.push_back(Sampler3D::sample_hemisphere(&seed));
      
    }
    for (uint i = 0; i < num_particles; i++) {
      Vector3f direction = o2w * Sampler3D::sample_hemisphere(&seed);
      float speed = Sampler1D::random(&seed) * max_starting_velocity;
      float temp = (2 * max_starting_velocity - speed) / (2 * max_starting_velocity) * 4000 +
                   (Sampler1D::random(&seed) * 500 - 250);
      particles.push_back(new Particle(starting_position, o2w * direction * speed, 1, .2, temp, 2000));
    }
  }

  void to_primitives(std::vector<Primitive *> *primitives) {
    for (Particle *p: particles) {
      Primitive *new_prim = new Primitive(PrimitiveFactory::createSphere(p->position, p->radius, p->new_bsdf()));
      primitives->push_back(new_prim);
    }
  }

  void step(float dt) {
    for (Particle *item: particles) {
      item->step(dt);
    }
  }
};