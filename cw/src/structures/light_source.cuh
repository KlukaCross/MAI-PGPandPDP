#pragma once

#include "vector3.cuh"

struct light_source_t {
    vec3d position;
    vec3f color;

    friend std::istream& operator>>(std::istream& in, light_source_t& light_source) {
        in >> light_source.position >> light_source.color;
        return in;
    }
};
