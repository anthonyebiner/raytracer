#pragma once

#include <utility>
#include "raytracer/util/misc.cuh"
#include "raytracer/scene/bbox.cuh"
#include "raytracer/scene/bsdf.cuh"
#include "raytracer/scene/primitives.cuh"
#include "Eigen/Dense"

using Eigen::Vector3f;

struct BVHNodeOpt {
  Vector3f minp;
  Vector3f maxp;

  union {
    struct {
      uint idx_left;
      uint idx_right;
    } inner;
    struct {
      uint count;
      uint idx_start;
    } leaf;
  };

  __device__ __host__ inline bool is_leaf() const {
    return leaf.count & 0x80000000;
  }
};


struct BVHAccelOpt {
  Primitive *primitives;
  BVHNodeOpt *nodes;
  uint num_primitives;
  uint num_nodes;

  BVHAccelOpt(Primitive *primitives, BVHNodeOpt *nodes, uint num_primitives, uint num_nodes)
      : primitives(primitives), nodes(nodes), num_primitives(num_primitives), num_nodes(num_nodes) {};

  ~BVHAccelOpt() {
    delete[] primitives;
    delete[] nodes;
  }

  __device__ __host__ bool intersect(Ray *ray) const {
    uint stack[MAX_BVH_STACK];
    uint stackIdx = 0;
    stack[stackIdx++] = 0;

    bool hit = false;

    while (stackIdx) {
      uint nodeIdx = stack[--stackIdx];
      BVHNodeOpt node = nodes[nodeIdx];

      // INTERSECT WITH BBOX
      if (!BBox::intersect(node.minp, node.maxp, ray)) continue;

      if (node.is_leaf()) {
        // INTERSECT WITH PRIMITIVES
        uint prim_count = node.leaf.count & 0x7fffffff;
        for (uint prim_idx = node.leaf.idx_start; prim_idx < node.leaf.idx_start + prim_count; prim_idx++) {
          bool h1 = primitives[prim_idx].intersect(ray);
          hit = h1 || hit;
        }
      } else {
        // PUSH LEFT AND RIGHT NODES TO STACK
        stack[stackIdx++] = node.inner.idx_left;
        stack[stackIdx++] = node.inner.idx_right;

        if (stackIdx > MAX_BVH_STACK) {
          printf("HIT MAX BVH STACK SIZE!!\n");
          return false;
        }
      }
    }

    return hit;
  }

  __host__ BVHAccelOpt *to_cuda() {
    BVHAccelOpt *bvh;

    cudaMalloc(&bvh, sizeof(BVHAccelOpt));
    cudaMemcpy(bvh, this, sizeof(BVHAccelOpt), cudaMemcpyHostToDevice);

    Primitive *d_primitives;
    cudaMalloc(&d_primitives, sizeof(Primitive) * num_primitives);
    cudaMemcpy(d_primitives, primitives, sizeof(Primitive) * num_primitives, cudaMemcpyHostToDevice);
    cudaMemcpy(&(bvh->primitives), &d_primitives, sizeof(Primitive *), cudaMemcpyHostToDevice);

    std::unordered_map<BSDF *, BSDF *> moved_bsdfs;
    for (uint i = 0; i < num_primitives; i++) {
      BSDF *bsdf = primitives[i].bsdf;
      if (bsdf == nullptr) continue;
      auto moved_bsdf = moved_bsdfs.find(bsdf);
      if (moved_bsdf != moved_bsdfs.end()) {
        cudaMemcpy(&(d_primitives[i].bsdf), &(moved_bsdf->second), sizeof(BSDF *), cudaMemcpyHostToDevice);
      } else {
        BSDF *d_bsdf;
        cudaMalloc(&d_bsdf, sizeof(BSDF));
        cudaMemcpy(d_bsdf, bsdf, sizeof(BSDF), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_primitives[i].bsdf), &d_bsdf, sizeof(BSDF *), cudaMemcpyHostToDevice);
        moved_bsdfs.insert({{bsdf, d_bsdf}});
      }
    }

    BVHNodeOpt *d_nodes;
    cudaMalloc(&d_nodes, sizeof(BVHNodeOpt) * num_nodes);
    cudaMemcpy(d_nodes, nodes, sizeof(BVHNodeOpt) * num_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(bvh->nodes), &d_nodes, sizeof(BVHNodeOpt *), cudaMemcpyHostToDevice);

    return bvh;
  }
};


struct BVHNode {
  BBox bb;
  BVHNode *l;
  BVHNode *r;

  std::vector<Primitive *>::const_iterator start;
  std::vector<Primitive *>::const_iterator end;

  explicit BVHNode(BBox bb) : bb(std::move(bb)), l(nullptr), r(nullptr) {}

  ~BVHNode() {
    delete l;
    delete r;
  }

  inline bool isLeaf() const { return l == nullptr && r == nullptr; }
};


class BVHAccel {
public:
  std::vector<Primitive *> primitives;
  BVHNode *root;

  explicit BVHAccel(const std::vector<Primitive *> &_primitives, uint max_leaf_size = 4) {
    primitives = std::vector<Primitive *>(_primitives);
    root = construct_bvh(primitives.begin(), primitives.end(), max_leaf_size);
  }

  ~BVHAccel() {
    delete root;
    primitives.clear();
  }

  BVHNode *construct_bvh(std::vector<Primitive *>::iterator start, std::vector<Primitive *>::iterator end,
                         uint max_leaf_size) {
    BBox bbox;
    for (auto p = start; p != end; p++) {
      auto bb = (*p)->get_bbox();
      bbox.expand((*p)->get_bbox());
    }

    auto node = new BVHNode(bbox);
    auto extent = bbox.extent;

    if (end - start <= max_leaf_size) {
      node->start = start;
      node->end = end;
      return node;
    } else {
      auto axis = int(std::distance(&extent[0], std::max_element(&extent[0], &extent[3])));

      auto center = Vector3f(0, 0, 0);
      for (auto p = start; p != end; p++) {
        center += (*p)->get_bbox().centroid();
      }
      center /= float(end - start);

      auto comp = [&](Primitive *i) -> bool {
        return i->get_bbox().centroid()[axis] < center[axis];
      };
      auto middle = partition(start, end, comp);

      if (middle - start == 0 || end - middle == 0) {
        node->start = start;
        node->end = end;
        return node;
      }
      node->l = construct_bvh(start, middle, max_leaf_size);
      node->r = construct_bvh(middle, end, max_leaf_size);
      return node;
    }
  }

  bool intersect(Ray *ray) const {
    return intersect(ray, root);
  }

  BVHAccelOpt to_optimized_bvh() const {
    uint num_primitives = primitives.size();
    uint num_nodes = get_num_nodes(root);

    auto prims = (Primitive *) malloc(num_primitives * sizeof(Primitive));
    auto nodes = (BVHNodeOpt *) malloc(num_nodes * sizeof(BVHNodeOpt));

    uint prim_start = 0;
    uint node_start = 0;

    to_optimized_bvh(root, prims, nodes, &prim_start, &node_start);

    return {prims, nodes, num_primitives, num_nodes};
  }

  uint get_num_nodes() const {
    return get_num_nodes(root);
  }

  uint get_max_depth() const {
    return get_max_depth(root, 0);
  }

  static bool intersect(Ray *ray, BVHNode *node) {
    if (!node->bb.intersect(ray)) {
      return false;
    }
    if (node->isLeaf()) {
      bool hit = false;
      for (auto p = node->start; p != node->end; p++) {
        bool h1 = (*p)->intersect(ray);
        hit = h1 || hit;
      }
      return hit;
    } else {
      bool h1 = intersect(ray, node->l);
      bool h2 = intersect(ray, node->r);
      return h1 || h2;
    }
  }

  static void to_optimized_bvh(BVHNode *node, Primitive *prims, BVHNodeOpt *nodes, uint *prim_start, uint *node_start) {
    uint n_idx = *node_start;

    nodes[n_idx].minp = node->bb.minp;
    nodes[n_idx].maxp = node->bb.maxp;
    if (node->isLeaf()) {
      nodes[n_idx].leaf.count = 0x80000000 | uint(node->end - node->start);
      nodes[n_idx].leaf.idx_start = *prim_start;
      for (auto p = node->start; p != node->end; p++) {
        prims[*prim_start] = **p;
        (*prim_start)++;
      }
    } else {
      nodes[n_idx].inner.idx_right = ++(*node_start);
      to_optimized_bvh(node->r, prims, nodes, prim_start, node_start);
      nodes[n_idx].inner.idx_left = ++(*node_start);
      to_optimized_bvh(node->l, prims, nodes, prim_start, node_start);
    }
  }

  static uint get_num_nodes(BVHNode *node) {
    if (node->isLeaf()) return 1;
    return 1 + get_num_nodes(node->l) + get_num_nodes(node->r);
  }

  static uint get_max_depth(BVHNode *node, uint depth) {
    if (node->isLeaf()) return depth;
    return max(get_max_depth(node->l, depth + 1), get_max_depth(node->r, depth + 1));
  }
};

