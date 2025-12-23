/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {

/// A structure which holds a combined Mesh.
/// While the MeshBlockLayer holds a mesh for each block,
/// this holds a single monolithic mesh for the entire map.
template <typename AppearanceType>
struct Mesh {
  /// Constructor
  Mesh(MemoryType memory_type = MemoryType::kDevice)
      : vertices(memory_type),
        vertex_normals(memory_type),
        triangles(memory_type),
        vertex_appearances(memory_type) {}

  /// Clear without deallocating
  void clearNoDeallocate() {
    vertices.clearNoDeallocate();
    vertex_normals.clearNoDeallocate();
    triangles.clearNoDeallocate();
    vertex_appearances.clearNoDeallocate();
  }

  // Data
  unified_vector<Vector3f> vertices;
  unified_vector<Vector3f> vertex_normals;
  unified_vector<int> triangles;
  unified_vector<AppearanceType> vertex_appearances;
};

using ColorMesh = Mesh<Color>;
using FeatureMesh = Mesh<FeatureArray>;

}  // namespace nvblox
