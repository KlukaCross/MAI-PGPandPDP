#pragma once

#include <assert.h>

#include "texture.cuh"
#include "triangle.cuh"
#include "ray.cuh"
#include "../variables/vars.cuh"

struct polygon_t {
    triangle_t trig;
    vec3f color;
    double a, b, c, d;
    float reflection_coef, transparent_coef, blend_coef;
    int light_sources_number;
    char textured;
    vec3d v1, v2, v3;
    texture_t tex;

    polygon_t() = delete;

    __host__ __device__ void buildPlane() {
        vec3d p0 = trig.a;
        vec3d edge1 = trig.b - p0;
        vec3d edge2 = trig.c - p0;
        a = edge1.y * edge2.z - edge1.z * edge2.y;
        b = -(edge1.x * edge2.z - edge1.z * edge2.x);
        c = edge1.x * edge2.y - edge1.y * edge2.x;
        d = -p0.x * a - p0.y * b - p0.z * c;
    }

    /* Glass */
    __host__ __device__ polygon_t(const triangle_t& _trig, const vec3f& _color, float _reflection_coef, float _transparent_coef)
        : trig(_trig),
          color(_color),
          reflection_coef(_reflection_coef),
          transparent_coef(_transparent_coef),
          blend_coef(1.0f - _reflection_coef - _transparent_coef),
          light_sources_number(),
          textured(),
          tex() {
        assert(blend_coef > -EPS);
        assert(blend_coef < 1 + EPS);
        buildPlane();
    }

    /* Corner */
    __host__ __device__ polygon_t(const triangle_t& _trig, const vec3f& _color)
        : trig(_trig),
          color(_color),
          reflection_coef(),
          transparent_coef(),
          blend_coef(1.0f),
          light_sources_number(),
          textured(),
          tex() {
        buildPlane();
    }

    /* Edge */
    __host__ __device__ polygon_t(const triangle_t& _trig, const vec3f& _color, int _light_sources_number, const vec3d& _v1, const vec3d& _v2)
        : trig(_trig),
          color(_color),
          reflection_coef(),
          transparent_coef(),
          blend_coef(1.0f),
          light_sources_number(_light_sources_number),
          textured(),
          v1(_v1),
          v2(_v2),
          tex() {
        assert(light_sources_number > 0);
        buildPlane();
    }

    /* Textured */
    __host__ __device__ polygon_t(const triangle_t& _trig, const vec3f& _color, float _reflection_coef, float _transparent_coef, const vec3d& _v1, const vec3d& _v2, const vec3d& _v3, const texture_t& _tex)
        : trig(_trig),
          color(_color),
          reflection_coef(_reflection_coef),
          transparent_coef(_transparent_coef),
          blend_coef(1.0f - _reflection_coef - _transparent_coef),
          light_sources_number(),
          textured(true),
          v1(_v1),
          v2(_v2),
          v3(_v3),
          tex(_tex) {
        assert(blend_coef > -EPS);
        assert(blend_coef < 1 + EPS);
        buildPlane();
    }


    __host__ __device__ vec3f getColor(const ray_t& ray, const vec3d& hit) const {
        if (textured) {
            vec3d p = hit - v3;
            double beta = (p.x * v1.y - p.y * v1.x) / (v2.x * v1.y - v2.y * v1.x);
            double alpha = (p.x * v2.y - p.y * v2.x) / (v1.x * v2.y - v1.y * v2.x);

            return tex.getPixel(alpha, beta);
        } else if (light_sources_number > 0 && vec3d::dot(trig.n, ray.v) > 0.0) {
            vec3d vl = (v2 - v1) / (light_sources_number + 1);

            for (int i = 1; i <= light_sources_number; ++i) {
                vec3d p_light = v1 + i * vl;

                if ((p_light - hit).length() < LIGHT_SIZE) {
                    return vec3f(EDGE_LIGHT, EDGE_LIGHT, EDGE_LIGHT);
                }
            }
        }
        return color;
    }

};
