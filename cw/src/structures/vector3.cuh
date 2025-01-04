#pragma once

#include <cmath>
#include <iostream>
#include "../variables/vars.cuh"


template <typename T>
struct vec3_t {
    T x, y, z;

    __host__ __device__ vec3_t() : x(0), y(0), z(0) {}
    __host__ __device__ vec3_t(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
    __host__ __device__ vec3_t(const vec3_t& v) : x(v.x), y(v.y), z(v.z) {}

    __host__ __device__ vec3_t& operator+=(const vec3_t& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    __host__ __device__ vec3_t& operator-=(const vec3_t& v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }

    __host__ __device__ vec3_t& operator*=(const vec3_t& v) {
        x *= v.x; y *= v.y; z *= v.z;
        return *this;
    }

    __host__ __device__ vec3_t& operator*=(T scalar) {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }

    __host__ __device__ vec3_t& operator/=(T scalar) {
        x /= scalar; y /= scalar; z /= scalar;
        return *this;
    }

    friend std::istream& operator>>(std::istream& in, vec3_t& v) {
        in >> v.x >> v.y >> v.z;
        return in;
    }

    __host__ __device__ double length() const {
        return std::sqrt(dot(*this, *this));
    }

    __host__ __device__ void normalize() {
        double len = length();
        x /= len; y /= len; z /= len;
    }

    __host__ __device__ static vec3_t fromCylindrical(T r, T z, T phi) {
        return {r * std::cos(phi), r * std::sin(phi), z};
    }

    __host__ __device__ static double dot(const vec3_t& a, const vec3_t& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ static vec3_t cross(const vec3_t& a, const vec3_t& b) {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    __host__ __device__ static vec3_t reflect(const vec3_t& l, const vec3_t& n) {
        vec3_t r = l - 2 * dot(n, l) * n;
        r.normalize();
        return r;
    }

    __host__ __device__ static float clamp(float value) {
        return (value > 1.0f) ? 1.0f : (value < 0.0f) ? 0.0f : value;
    }

    __host__ __device__ void clamp() {
        x = clamp(x); y = clamp(y); z = clamp(z);
    }

    __device__ static void atomicAddVec(vec3_t* a, const vec3_t& b) {
        atomicAdd(&(a->x), b.x);
        atomicAdd(&(a->y), b.y);
        atomicAdd(&(a->z), b.z);
    }

    friend __host__ __device__ vec3_t operator+(vec3_t a, const vec3_t& b) { return a += b; }
    friend __host__ __device__ vec3_t operator-(vec3_t a, const vec3_t& b) { return a -= b; }
    friend __host__ __device__ vec3_t operator*(vec3_t a, const vec3_t& b) { return a *= b; }
    friend __host__ __device__ vec3_t operator*(T scalar, vec3_t v) { return v *= scalar; }
    friend __host__ __device__ vec3_t operator/(vec3_t v, T scalar) { return v /= scalar; }
};


using vec3f = vec3_t<float>;
using vec3d = vec3_t<double>;
using vec3i = vec3_t<int>;
