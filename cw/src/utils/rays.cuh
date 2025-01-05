#pragma once

#include "../structures/ray.cuh"
#include "../structures/polygon.cuh"

__host__ __device__ void intersectRayPlane(const ray_t& ray, const polygon_t& polygon, double& t) {
    t = -(polygon.a * ray.p.x + polygon.b * ray.p.y + polygon.c * ray.p.z + polygon.d) / (polygon.a * ray.v.x + polygon.b * ray.v.y + polygon.c * ray.v.z);
}


__host__ __device__ void intersectRayPolygon(const ray_t& ray, const polygon_t& polygon, double& t, bool& ans) {
    const vec3d P = vec3d::cross(ray.v, polygon.trig.e2);
    const double div = vec3d::dot(P, polygon.trig.e1);

    // Если делитель почти нулевой, луч и полигон параллельны
    if (fabs(div) < EPS) {
        ans = false;
        return;
    }

    const vec3d T = ray.p - polygon.trig.a;
    const double u = vec3d::dot(P, T) / div;
    if (u < 0.0 || u > 1.0) {
        ans = false;
        return;
    }

    const vec3d Q = vec3d::cross(T, polygon.trig.e1);
    const double v = vec3d::dot(Q, ray.v) / div;
    if (v < 0.0 || u + v > 1.0) {
        ans = false;
        return;
    }

    t = vec3d::dot(Q, polygon.trig.e2) / div;
    ans = (t >= 0.0);
}
