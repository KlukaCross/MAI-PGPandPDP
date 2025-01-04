#pragma once

#include "../scene/shading.cuh"
#include "../utils/cuda.cuh"
#include "../utils/print.cuh"
#include "../structures/polygon.cuh"
#include "../structures/ray.cuh"
#include "../structures/vector3.cuh"
#include "../structures/light_source.cuh"

__global__ void gpuClearData(vec3f* data_dev, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += offset) {
        data_dev[i] = vec3f(0, 0, 0);
    }
}


__device__ void gpuUpdateRayOut(ray_t* rays_out, int* size_out, const ray_t& ray, const vec3d& direction, const vec3d& hit_point, double coef, const vec3f& color) {
    rays_out[atomicAdd(size_out, 1)] = ray_t(hit_point + POLY_COEF * direction, direction, ray.pix_id, coef * ray.coef * color);
}

__device__ void gpuTraceRay(const ray_t& ray, int polygons_number, const polygon_t* polygons, int& min_i, double& min_t) {
    for (int i = 0; i < polygons_number; i++) {
        double t;
        bool flag;
        intersectRayPolygon(ray, polygons[i], t, flag);
        if (flag && t < min_t) {
            min_i = i;
            min_t = t;
        }
    }
}

__global__ void gpuTrace(const ray_t* rays_in, int size_in, ray_t* rays_out, int* size_out, vec3f* data_dev, const light_source_t* light_sources_dev, int n_sources, const polygon_t* polygons_dev, int polygons_number) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;

    for (int k = idx; k < size_in; k += offset) {
        int min_i = polygons_number;
        double min_t = INF;

        // Find the closest polygon intersection
        gpuTraceRay(rays_in[k], polygons_number, polygons_dev, min_i, min_t);

        if (min_i == polygons_number) continue;

        vec3d hit = rays_in[k].p + min_t * rays_in[k].v;
        vec3f poly_color = polygons_dev[min_i].getColor(rays_in[k], hit);

        // Add Phong shading
        vec3f::atomicAddVec(&data_dev[rays_in[k].pix_id], phongShading(rays_in[k], hit, polygons_dev[min_i], min_i, light_sources_dev, n_sources, polygons_dev, polygons_number));

        // Handle transparency and reflection
        const polygon_t& min_polygon = polygons_dev[min_i];
        if (min_polygon.transparent_coef > 0) {
            gpuUpdateRayOut(rays_out, size_out, rays_in[k], rays_in[k].v, hit, min_polygon.transparent_coef, poly_color);
        }
        if (min_polygon.reflection_coef > 0) {
            vec3d reflected = vec3d::reflect(rays_in[k].v, min_polygon.trig.n);
            gpuUpdateRayOut(rays_out, size_out, rays_in[k], reflected, hit, min_polygon.reflection_coef, poly_color);
        }
    }
}


__global__ void gpuInitRays(const vec3d camera_pc, const vec3d camera_pv, int w, int h, double angle, ray_t* rays_dev) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / std::tan(angle * M_PI / 360.0);

    vec3d bz = camera_pv - camera_pc;
    vec3d bx = vec3d::cross(bz, vec3d(0, 0, 1));
    vec3d by = vec3d::cross(bx, bz);
    normalizeVectors(bx, by, bz);

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < w; i += offsetx) {
        for (int j = idy; j < h; j += offsety) {
            vec3d v(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            rays_dev[i * h + j] = ray_t(camera_pc, mult(bx, by, bz, v), (h - 1 - j) * w + i);
        }
    }
}

__global__ void gpuWriteData(uchar4* data_dev, vec3f* data_dev_vec3f, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += offset) {
        data_dev_vec3f[i].clamp();
        data_dev_vec3f[i] *= 255.0f;
        data_dev[i] = make_uchar4(data_dev_vec3f[i].x, data_dev_vec3f[i].y, data_dev_vec3f[i].z, 255);
    }
}


void gpuRender(int frame_id, const vec3d camera_pc, const vec3d camera_pv, int w, int h, double angle, uchar4* data_dev, light_source_t* light_sources_dev, polygon_t* polygons_dev, int light_sources_number, int polygons_number, int recursion_depth) {
    int size_in = w * h;
    vec3f* data_dev_vec3f;
    CSC(cudaMalloc(&data_dev_vec3f, sizeof(vec3f) * size_in));
    gpuClearData<<<BLOCKS, THREADS>>>(data_dev_vec3f, size_in);

    ray_t* ray_in_dev;
    CSC(cudaMalloc(&ray_in_dev, sizeof(ray_t) * size_in));
    gpuInitRays<<<BLOCKS_2D, THREADS_2D>>>(camera_pc, camera_pv, w, h, angle, ray_in_dev);

    long long total_rays = 0;
    cuda_timer_t tmr;
    tmr.start();

    for (int rec = 0; rec < recursion_depth && size_in; ++rec) {
        total_rays += size_in;

        ray_t* ray_out_dev;
        CSC(cudaMalloc(&ray_out_dev, 2 * sizeof(ray_t) * size_in));

        int zero = 0;
        int* size_out;
        CSC(cudaMalloc(&size_out, sizeof(int)));
        CSC(cudaMemcpy(size_out, &zero, sizeof(int), cudaMemcpyHostToDevice));

        gpuTrace<<<BLOCKS, THREADS>>>(ray_in_dev, size_in, ray_out_dev, size_out, data_dev_vec3f,
                                      light_sources_dev, light_sources_number, polygons_dev, polygons_number);

        CSC(cudaFree(ray_in_dev));
        ray_in_dev = ray_out_dev;

        CSC(cudaMemcpy(&size_in, size_out, sizeof(int), cudaMemcpyDeviceToHost));
        CSC(cudaFree(size_out));
    }

    gpuWriteData<<<BLOCKS, THREADS>>>(data_dev, data_dev_vec3f, w * h);
    tmr.end();

    CSC(cudaFree(ray_in_dev));
    CSC(cudaFree(data_dev_vec3f));

    print(frame_id, tmr.get_time(), total_rays);
    fflush(stdout);
}



