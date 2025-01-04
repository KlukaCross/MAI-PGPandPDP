#pragma once

#include <assert.h>
#include <algorithm>
#include <set>
#include <iostream>

#include "../utils/rays.cuh"
#include "objects/base.cuh"
#include "objects/dodecahedron.cuh"
#include "objects/icosahedron.cuh"
#include "objects/octahedron.cuh"
#include "../structures/floor.cuh"
#include "../structures/triangle.cuh"
#include "../structures/polygon.cuh"
#include "../structures/figure.cuh"
#include "../variables/vars.cuh"
#include "../variables/gpu_vars.cuh"

const vec3f EDGE_COLOR(EDGE_COLOR_RED, EDGE_COLOR_GREEN, EDGE_COLOR_BLUE);

void buildObject(int id, const base_object_t& obj, std::vector<figure_t>& figures, std::vector<polygon_t>& polygons) {
    std::vector<vec3d> vertices;
    std::vector<std::set<int>> vertex_polygons;
    std::vector<vec3i> polygon_indexes;

    for (int i = 0; i < obj.vertices.size(); ++i) {
        vec3d vertex = obj.vertices[i];
        vertex *= figures[id].radius;
        vertices.push_back(vertex);
        vertex_polygons.push_back(std::set<int>());
    }
    for (int i = 0; i < obj.polygons.size(); ++i) {
        vec3i ids = obj.polygons[i];
        --ids.x;
        --ids.y;
        --ids.z;
        polygon_indexes.push_back(ids);
        vertex_polygons[ids.x].insert(i);
        vertex_polygons[ids.y].insert(i);
        vertex_polygons[ids.z].insert(i);
    }

    double side = INF;
    int m = vertices.size();
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            side = std::min(side, (vertices[i] - vertices[j]).length());
        }
    }

    std::set<int> unique_polygon_ids;
    for (int i = 0; i < m; ++i) {
        vec3d vi = vertices[i];
        for (int j = i + 1; j < m; ++j) {
            vec3d vj = vertices[j];
            if ((vi - vj).length() > side + EPS) {
                continue;
            }
            std::vector<int> trig_ids;
            std::vector<triangle_t> trigs;
            for (int elem : vertex_polygons[i]) {
                if (vertex_polygons[j].count(elem)) {
                    trig_ids.push_back(elem);
                    vec3i ids = polygon_indexes[elem];
                    trigs.push_back(triangle_t(vertices[ids.x], vertices[ids.y], vertices[ids.z]));
                }
            }
            assert(trigs.size() == 2);

            vec3d n1 = EDGE_THICKNESS * trigs[0].n;
            vec3d n2 = EDGE_THICKNESS * trigs[1].n;
            vec3d n_avg = (n1 + n2) / 2;

            trigs[0].shift(n1);
            trigs[1].shift(n2);

            vec3d vi1 = vi + n1;
            vec3d vi2 = vi + n2;
            vec3d vj1 = vj + n1;
            vec3d vj2 = vj + n2;
            vec3d vi_avg = (vi1 + vi2) / 2 + figures[id].center;
            vec3d vj_avg = (vj1 + vj2) / 2 + figures[id].center;

            double t;
            triangle_t edge_1(vi1, vj2, vi2);
            edge_1.validateNormal(n_avg);
            intersectRayPlane(ray_t(vec3d(0, 0, 0), vi, -1), polygon_t(edge_1, EDGE_COLOR), t);
            triangle_t corner_1(vi1, vi2, t * vi / vi.length());
            corner_1.validateNormal(n_avg);
            edge_1.shift(figures[id].center);
            corner_1.shift(figures[id].center);
            polygons.push_back(polygon_t(edge_1, EDGE_COLOR, figures[id].light_sources_number, vi_avg, vj_avg));
            polygons.push_back(polygon_t(corner_1, EDGE_COLOR));

            triangle_t edge_2(vi1, vj1, vj2);
            edge_2.validateNormal(n_avg);
            intersectRayPlane(ray_t(vec3d(0, 0, 0), vj, -1), polygon_t(edge_2, EDGE_COLOR), t);
            triangle_t corner_2(vj1, t * vj / vj.length(), vj2);
            corner_2.validateNormal(n_avg);
            edge_2.shift(figures[id].center);
            corner_2.shift(figures[id].center);
            polygons.push_back(polygon_t(edge_2, EDGE_COLOR, figures[id].light_sources_number, vi_avg, vj_avg));
            polygons.push_back(polygon_t(corner_2, EDGE_COLOR));

            for (int k = 0; k < 2; ++k) {
                if (!unique_polygon_ids.count(trig_ids[k])) {
                    trigs[k].shift(figures[id].center);
                    polygons.push_back(polygon_t(trigs[k], figures[id].color, figures[id].reflection_coef, figures[id].transparent_coef));
                    unique_polygon_ids.insert(trig_ids[k]);
                }
            }
        }
    }
    assert(unique_polygon_ids.size() == polygon_indexes.size());
}

void buildScene(bool is_used_gpu, floor_t& floor, std::vector<polygon_t>& polygons, std::vector<figure_t>& figures) {
    floor.texture.load(floor.texture_path, is_used_gpu);
    triangle_t t1 = {floor.points[0], floor.points[2], floor.points[1]};
    triangle_t t2 = {floor.points[2], floor.points[0], floor.points[3]};
    polygons.push_back(polygon_t(t1, floor.color, floor.reflection, 0.0, floor.points[1] - floor.points[2], floor.points[1] - floor.points[0], floor.points[0] + floor.points[2] - floor.points[1], floor.texture));
    polygons.push_back(polygon_t(t2, floor.color, floor.reflection, 0.0, floor.points[0] - floor.points[3], floor.points[2] - floor.points[3], floor.points[3], floor.texture));
    buildObject(0, octahedron_obj, figures, polygons);
    buildObject(1, dodecahedron_obj, figures, polygons);
    buildObject(2, icosahedron_obj, figures, polygons);
}
