#pragma once

#include "vector3.cuh"


struct triangle_t {
    vec3d a, b, c, n, e1, e2;

    triangle_t() = delete;

    __host__ __device__ triangle_t(const vec3d& _a, const vec3d& _b, const vec3d& _c) : a(_a), b(_b), c(_c), e1(b - a), e2(c - a) {
        n = vec3d::cross(e1, e2);
        n.normalize();
    }

    __host__ __device__ void shift(const vec3d& v) {
        a += v;
        b += v;
        c += v;
    }

    void validateNormal(const vec3d& _n) {
        if (vec3d::dot(_n, n) < -EPS) {
            std::swap(a, c);
            e1 = b - a;
            e2 = c - a;
            n = vec3d::cross(e1, e2);
            n.normalize();
        }
    }
};

