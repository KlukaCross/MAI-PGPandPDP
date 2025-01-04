#pragma once

#include "../structures/ray.cuh"
#include "../structures/polygon.cuh"

__host__ __device__ void intersectRayPlane(const ray_t& r, const polygon_t& poly, double& t) {
    t = -(poly.a * r.p.x + poly.b * r.p.y + poly.c * r.p.z + poly.d) / (poly.a * r.v.x + poly.b * r.v.y + poly.c * r.v.z);
}


__host__ __device__ void intersectRayPolygon(const ray_t& r, const polygon_t& poly, double& t, bool& ans) {
    const vec3d P = vec3d::cross(r.v, poly.trig.e2);
    const double div = vec3d::dot(P, poly.trig.e1);

    // Если делитель почти нулевой, луч и полигон параллельны
    if (fabs(div) < EPS) {
        ans = false;
        return;
    }

    const vec3d T = r.p - poly.trig.a;
    const double u = vec3d::dot(P, T) / div;
    if (u < 0.0 || u > 1.0) {
        ans = false;
        return;
    }

    const vec3d Q = vec3d::cross(T, poly.trig.e1);
    const double v = vec3d::dot(Q, r.v) / div;
    if (v < 0.0 || u + v > 1.0) {
        ans = false;
        return;
    }

    t = vec3d::dot(Q, poly.trig.e2) / div;
    ans = (t >= 0.0);
}
