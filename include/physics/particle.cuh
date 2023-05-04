#pragma once

#include "raytracer/linalg/Vector3f.cuh"
#include "raytracer/scene/bsdf.cuh"

#define GRAVITY_ACCEL -9.8
#define DRAG_COEF 0.45

static Vector3f kelvinToRGB(float k) {
  float temp = k / 100;
  float red, green, blue;
  if (temp <= 66) {
    red = 61 * log(temp);
    green = temp;
    green = 99.4708025861 * log(green) - 161.1195681661;
    if (temp <= 19) {
      blue = 0;
    } else {
      blue = temp - 10;
      blue = 138.5177312231 * log(blue) - 305.0447927307;
    }
  } else {
    red = temp - 60;
    red = 329.698727446 * pow(red, -0.1332047592);
    green = temp - 60;
    green = 288.1221695283 * pow(green, -0.0755148492);
    blue = 255;
  }
  return {
      clamp(red, 0.f, 255.f) / 255,
      clamp(green, 0.f, 255.f) / 255,
      clamp(blue, 0.f, 255.f) / 255
  };
}


class Particle {
public:
  Vector3f position;
  Vector3f velocity;
  float mass;
  float radius;
  float temp;
  float temp_decay;

  Particle(const Vector3f &starting_position, const Vector3f &starting_velocity, float starting_mass,
           float starting_radius, float starting_temp, float starting_temp_decay) {
    position = starting_position;
    velocity = starting_velocity;
    mass = starting_mass;
    radius = starting_radius;
    temp = starting_temp;
    temp_decay = starting_temp_decay;
  }

  BSDF *new_bsdf() {
    return new BSDF(BSDFFactory::createEmission(kelvinToRGB(temp)));
  }

  Vector3f get_drag() {
    Vector3f drag = DRAG_COEF * pow(radius, 2) * velocity * velocity.norm();
    return drag;
  }

  Vector3f get_forces(float dt) {
    return Vector3f(0, 1, 0) * GRAVITY_ACCEL * mass - get_drag();
  }

  void clamp_position() {
    if (position.y < 0) {
      position.y = 0;
      velocity.y = 0;
    }
  }

  void clamp_temperature() {
    temp = maxf(temp, 100);
  }

  void step_position(float dt) {
    Vector3f forces = get_forces(dt);
    Vector3f acceleration = forces / mass;
    position += velocity * dt + acceleration * pow(dt, 2) / 2;
    velocity += acceleration * dt;
    clamp_position();
  }

  void step_temperature(float dt) {
    temp -= temp_decay * dt;
    clamp_temperature();
  }

  void step(float dt) {
    step_position(dt);
    step_temperature(dt);
  }
};