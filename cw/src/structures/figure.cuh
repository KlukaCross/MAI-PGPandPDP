#pragma once

struct figure_t {
    vec3d center;
    vec3f color;
    double radius;
    double reflection_coef, transparent_coef, light_sources_number;

    friend std::istream& operator>>(std::istream& in, figure_t& figure) {
        in >> figure.center >> figure.color >> figure.radius >> figure.reflection_coef >> figure.transparent_coef >> figure.light_sources_number;
        return in;
    }
};
