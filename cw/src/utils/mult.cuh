#pragma once

__host__ __device__ vec3d mult(const vec3d& a, const vec3d& b, const vec3d& c, const vec3d& v) {
    return {a.x * v.x + b.x * v.y + c.x * v.z,
            a.y * v.x + b.y * v.y + c.y * v.z,
            a.z * v.x + b.z * v.y + c.z * v.z};
}

void normalizeVectors(vec3d& bx, vec3d& by, vec3d& bz) {
    bx.normalize();
    by.normalize();
    bz.normalize();
}
