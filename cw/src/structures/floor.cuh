#pragma once

#include "vector3.cuh"
#include "texture.cuh"

const int FLOOR_POINTS_NUMBER = 4;

struct floor_t {
    std::vector<vec3d> points;
    std::string texture_path;
    vec3f color;
    double reflection;

    texture_t texture;

    floor_t() : points(FLOOR_POINTS_NUMBER) {}

    friend std::istream& operator>>(std::istream& in, floor_t& floor) {
        for (int i = 0; i < FLOOR_POINTS_NUMBER; ++i) {
            in >> floor.points[i];
        }
        in >> floor.texture_path >> floor.color >> floor.reflection;
        return in;
    }
};
