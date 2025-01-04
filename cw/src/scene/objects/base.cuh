#pragma once

#include <vector>

#include "../../structures/vector3.cuh"

struct base_object_t {
    std::vector<vec3d> vertices;
    std::vector<vec3i> polygons;

    base_object_t(const std::vector<vec3d>& _vertices, const std::vector<vec3i>& _polygons) : vertices(_vertices), polygons(_polygons) {};
};

