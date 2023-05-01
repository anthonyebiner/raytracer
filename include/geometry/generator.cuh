#pragma once

#include "rapidobj/rapidobj.hpp"
#include "raytracer/scene/primitives.cuh"
#include "raytracer/scene/bsdf.cuh"
#include "Eigen/Dense"

using Eigen::Vector3f;


struct Generator {
  static bool
  room(std::vector<Primitive *> *primitives, float x_scale, float y_scale, float z_scale,
       BSDF *floor_bsdf, BSDF *ceiling_bsdf, BSDF *left_bsdf, BSDF *right_bsdf, BSDF *front_bsdf, BSDF *back_bsdf) {
    // Floor
    if (floor_bsdf != nullptr) {
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(-x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(x_scale / 2, 0, z_scale / 2),
                                           Vector3f(0, 1, 0), Vector3f(0, 1, 0), Vector3f(0, 1, 0), floor_bsdf)));
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(-x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(-x_scale / 2, 0, z_scale / 2),
                                           Vector3f(x_scale / 2, 0, z_scale / 2),
                                           Vector3f(0, 1, 0), Vector3f(0, 1, 0), Vector3f(0, 1, 0), floor_bsdf)));
    }

    // Ceiling
    if (ceiling_bsdf != nullptr) {
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(-x_scale / 2, y_scale, -z_scale / 2),
                                           Vector3f(x_scale / 2, y_scale, -z_scale / 2),
                                           Vector3f(x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), ceiling_bsdf)));
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(-x_scale / 2, y_scale, -z_scale / 2),
                                           Vector3f(-x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(0, -1, 0), Vector3f(0, -1, 0), Vector3f(0, -1, 0), ceiling_bsdf)));
    }

    // Left wall
    if (left_bsdf != nullptr) {
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(-x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(-x_scale / 2, y_scale, -z_scale / 2),
                                           Vector3f(-x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(1, 0, 0), Vector3f(1, 0, 0), Vector3f(1, 0, 0), left_bsdf)));
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(-x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(-x_scale / 2, 0, z_scale / 2),
                                           Vector3f(-x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(1, 0, 0), Vector3f(1, 0, 0), Vector3f(1, 0, 0), left_bsdf)));
    }

    // Right wall
    if (right_bsdf != nullptr) {
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(x_scale / 2, y_scale, -z_scale / 2),
                                           Vector3f(x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), right_bsdf)));
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(x_scale / 2, 0, z_scale / 2),
                                           Vector3f(x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), Vector3f(-1, 0, 0), right_bsdf)));
    }

    // Back wall
    if (back_bsdf != nullptr) {
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(x_scale / 2, 0, z_scale / 2),
                                           Vector3f(-x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(0, 0, -1), Vector3f(0, 0, -1), Vector3f(0, 0, -1), back_bsdf)));
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(x_scale / 2, 0, z_scale / 2),
                                           Vector3f(-x_scale / 2, y_scale, z_scale / 2),
                                           Vector3f(-x_scale / 2, 0, z_scale / 2),
                                           Vector3f(0, 0, -1), Vector3f(0, 0, -1), Vector3f(0, 0, -1), back_bsdf)));
    }

    // Front wall
    if (front_bsdf != nullptr) {
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(-x_scale / 2, y_scale, -z_scale / 2),
                                           Vector3f(x_scale / 2, y_scale, -z_scale / 2),
                                           Vector3f(0, 0, 1), Vector3f(0, 0, 1), Vector3f(0, 0, 1), front_bsdf)));
      primitives->push_back(new Primitive(
          PrimitiveFactory::createTriangle(Vector3f(x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(-x_scale / 2, y_scale, -z_scale / 2),
                                           Vector3f(-x_scale / 2, 0, -z_scale / 2),
                                           Vector3f(0, 0, 1), Vector3f(0, 0, 1), Vector3f(0, 0, 1), front_bsdf)));
    }

  }

  static bool
  from_obj(std::vector<Primitive *> *primitives, std::string file, BSDF *bsdf = nullptr, Vector3f scale = {1, 1, 1},
           Vector3f translate = {0, 0, 0}) {
    rapidobj::Result result = rapidobj::ParseFile(file);

    if (result.error) {
      print("{}\n", result.error.code.message());
      return false;
    }

    bool success = rapidobj::Triangulate(result);

    if (!success) {
      print("Unable to triangulate\n");
      return false;
    }

    BSDF *new_bsdf;
    std::unordered_map<int, BSDF *> bsdfs;
    for (const auto &shape: result.shapes) {
      for (uint i = 0; i < shape.mesh.indices.size(); i += 3) {
        Vector3f vertices[3];
        Vector3f normals[3];
        auto m_id = shape.mesh.material_ids[i / 3];
        if (bsdf == nullptr && m_id != -1) {
          auto bsdf_entry = bsdfs.find(m_id);
          if (bsdf_entry == bsdfs.end()) {
            auto material = result.materials[m_id];
            if (material.emission[0] > 0) {
              new_bsdf = new BSDF(
                  BSDFFactory::createEmission({material.emission[0], material.emission[1], material.emission[2]}));
            } else if (material.illum == 3) {
              new_bsdf = &mirror_bsdf;
            } else if (material.illum == 4) {
              Array3f transmittance = {material.transmittance[0], material.transmittance[1],
                                       material.transmittance[2]};
              new_bsdf = new BSDF(
                  BSDFFactory::createGlass(material.ior, 0, transmittance, transmittance));
            } else {
              Array3f diffuse = {material.diffuse[0], material.diffuse[1], material.diffuse[2]};
              Array3f specular = {material.specular[0], material.specular[1], material.specular[2]};

              new_bsdf = new BSDF(
                  BSDFFactory::createPhong(material.shininess, diffuse, specular));
            }
            bsdfs.insert({m_id, new_bsdf});
          } else {
            new_bsdf = bsdf_entry->second;
          }
        } else {
          new_bsdf = bsdf;
        }

        for (uint j = 0; j < 3; j++) {
          auto index = shape.mesh.indices[i + j];
          vertices[j] = {
              result.attributes.positions[index.position_index * 3 + 0],
              result.attributes.positions[index.position_index * 3 + 1],
              result.attributes.positions[index.position_index * 3 + 2]
          };
          normals[j] = {
              result.attributes.normals[index.normal_index * 3 + 0],
              result.attributes.normals[index.normal_index * 3 + 1],
              result.attributes.normals[index.normal_index * 3 + 2]
          };
          vertices[j] = Vector3f(vertices[j].array() * scale.array());
          vertices[j] += translate;
        }


        primitives->push_back(new Primitive(
            PrimitiveFactory::createTriangle(vertices[0], vertices[1], vertices[2], normals[0], normals[1], normals[2],
                                             new_bsdf)));
      }
    }
    return true;
  }
};
