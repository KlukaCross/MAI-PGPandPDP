#pragma once

#include "../structures/ray.cuh"
#include "../structures/vector3.cuh"
#include "../structures/polygon.cuh"
#include "../structures/light_source.cuh"
#include "../utils/rays.cuh"

#define AMBIENT_COEF 0.25
#define SOURCE_COEF 1.0
#define DIFFUSE_COEF 1.0
#define SPECULAR_COEF 0.5

__host__ __device__ vec3f phongShading(const ray_t& ray, const vec3d& hit, const polygon_t& polygon, int id, const light_source_t* light_sources, int ligth_sources_number, const polygon_t* polygons, int n_polygons) {
    vec3f polygon_color = polygon.getColor(ray, hit);
    vec3f resulting_color = AMBIENT_COEF * polygon.blend_coef * ray.coef * polygon_color;

    for (int j = 0; j < ligth_sources_number; ++j) {
        vec3d light_dir = light_sources[j].position - hit;
        double t_max = light_dir.length();
        ray_t light_ray(hit, light_dir, ray.pix_id);
        vec3f visibility_coef(1.0f, 1.0f, 1.0f);

        for (int i = 0; i < n_polygons; ++i) {
            if (i == id) continue;

            double t;
            bool flag;
            intersectRayPolygon(light_ray, polygons[i], t, flag);
            if (flag && t < t_max) {
                visibility_coef *= polygons[i].transparent_coef;
            }
        }

        vec3f light_contrib = polygon.blend_coef * ray.coef * visibility_coef * light_sources[j].color * polygon_color;
        double diffuse_coef = max(0.0, vec3d::dot(polygon.trig.n, light_ray.v));
        double specular_coef = 0.0;

        if (diffuse_coef > 0.0) {
            vec3d reflected = vec3d::reflect(light_ray.v, polygon.trig.n);
            specular_coef = max(0.0, vec3d::dot(reflected, ray.v));
            specular_coef = std::pow(specular_coef, 9);
        }

        resulting_color += SOURCE_COEF * (DIFFUSE_COEF * diffuse_coef + SPECULAR_COEF * specular_coef) * light_contrib;
    }

    resulting_color.clamp();
    return resulting_color;
}
