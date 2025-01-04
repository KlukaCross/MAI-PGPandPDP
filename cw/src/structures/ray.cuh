#pragma once

#include "vector3.cuh"

struct ray_t {
    vec3d p, v;
    int pix_id;
    vec3f coef;

    __host__ __device__ ray_t() : ray_t(vec3d(), vec3d(), 0, vec3f(1.0f, 1.0f, 1.0f)) {}

    __host__ __device__ ray_t(const vec3d& _p, const vec3d& _v, int _pix_id, const vec3f& _coef = vec3f(1.0f, 1.0f, 1.0f)) : p(_p), v(_v), pix_id(_pix_id), coef(_coef) {
        v.normalize();
    }
};
