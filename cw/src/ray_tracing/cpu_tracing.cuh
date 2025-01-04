#pragma once

#include "../scene/shading.cuh"
#include "../utils/rays.cuh"
#include "../utils/vector.cuh"
#include "../utils/print.cuh"
#include "../structures/polygon.cuh"
#include "../structures/ray.cuh"
#include "../structures/vector3.cuh"
#include "../structures/light_source.cuh"

void cpuClearData(vec3f* data, const int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = vec3f(0, 0, 0);
    }
}


void cpuFindClosestIntersection(const ray_t& ray, const std::vector<polygon_t>& polygons, int& min_i, double& min_t) {
    min_i = polygons.size();
    min_t = INF;
    for (int i = 0; i < polygons.size(); ++i) {
        double t;
        bool flag;
        intersectRayPolygon(ray, polygons[i], t, flag);
        if (flag && t < min_t) {
            min_i = i;
            min_t = t;
        }
    }
}

void cpuHandleRayOutput(const ray_t& ray_in, const vec3d& hit, const polygon_t& polygon, double coef, vec3d direction, ray_t* rays_out, int& size_out, vec3f& color) {
    rays_out[size_out++] = ray_t(hit + POLY_COEF * direction, direction, ray_in.pix_id, coef * ray_in.coef * color);
}

void cpuTrace(const ray_t* rays_in, int size_in, ray_t* rays_out, int& size_out, vec3f* data, std::vector<polygon_t>& polygons, std::vector<light_source_t>& light_sources) {
    for (int k = 0; k < size_in; ++k) {
        int min_i;
        double min_t;
        cpuFindClosestIntersection(rays_in[k], polygons, min_i, min_t);

        if (min_i == polygons.size()) continue;

        const polygon_t& polygon = polygons[min_i];
        vec3d hit = rays_in[k].p + min_t * rays_in[k].v;
        vec3f color = polygon.getColor(rays_in[k], hit);

        // Add phong shading
        data[rays_in[k].pix_id] += phongShading(rays_in[k], hit, polygon, min_i, light_sources.data(), light_sources.size(), polygons.data(), polygons.size());

        // Handle transparency
        if (polygon.transparent_coef > 0) {
            cpuHandleRayOutput(rays_in[k], hit, polygon, polygon.transparent_coef, rays_in[k].v, rays_out, size_out, color);
        }

        // Handle reflection
        if (polygon.reflection_coef > 0) {
            vec3d reflected = vec3d::reflect(rays_in[k].v, polygon.trig.n);
            cpuHandleRayOutput(rays_in[k], hit, polygon, polygon.reflection_coef, reflected, rays_out, size_out, color);
        }
    }
}


void cpuInitRays(const vec3d& camera_pc, const vec3d& camera_pv, int w, int h, double angle, ray_t* rays) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / std::tan(angle * M_PI / 360.0);

    vec3d bz = camera_pv - camera_pc;
    vec3d bx = vec3d::cross(bz, vec3d(0, 0, 1));
    vec3d by = vec3d::cross(bx, bz);
    normalizeVectors(bx, by, bz);

    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            vec3d v(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            rays[i * h + j] = ray_t(camera_pc, mult(bx, by, bz, v), (h - 1 - j) * w + i);
        }
    }
}

void cpuWriteData(uchar4* data, vec3f* data_vec3f, int sz) {
    for (int i = 0; i < sz; ++i) {
        data_vec3f[i].clamp();
        data_vec3f[i] *= 255.0f;
        data[i] = make_uchar4(data_vec3f[i].x, data_vec3f[i].y, data_vec3f[i].z, 255);
    }
}


void cpuRender(int frame_id, const vec3d& camera_pc, const vec3d& camera_pv, int w, int h, double angle, uchar4* data, std::vector<polygon_t>& polygons, std::vector<light_source_t>& light_sources, int recursion_depth) {
    int size_in = w * h;
    std::vector<vec3f> data_vec3f(size_in);
    cpuClearData(data_vec3f.data(), size_in);

    std::vector<ray_t> ray_in(size_in);
    cpuInitRays(camera_pc, camera_pv, w, h, angle, ray_in.data());

    long long total_rays = 0;
    cuda_timer_t tmr;
    tmr.start();

    for (int rec = 0; rec < recursion_depth && size_in; ++rec) {
        total_rays += size_in;

        std::vector<ray_t> ray_out(2 * size_in);
        int size_out = 0;
        cpuTrace(ray_in.data(), size_in, ray_out.data(), size_out, data_vec3f.data(), polygons, light_sources);

        ray_in.swap(ray_out);
        size_in = size_out;
    }

    cpuWriteData(data, data_vec3f.data(), w * h);
    tmr.end();

    print(frame_id, tmr.get_time(), total_rays);
}

