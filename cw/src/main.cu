#include <math.h>
#include <stdlib.h>

#include "scene/build.cuh"
#include "utils/image_io.cuh"
#include "utils/cuda.cuh"
#include "ray_tracing/cpu_tracing.cuh"
#include "ray_tracing/gpu_tracing.cuh"
#include "ssaa/cpu_ssaa.cuh"
#include "ssaa/gpu_ssaa.cuh"
#include "structures/camera.cuh"
#include "structures/figure.cuh"
#include "structures/floor.cuh"
#include "structures/polygon.cuh"
#include "structures/vector3.cuh"
#include "structures/light_source.cuh"

const int FIGURES_NUMBER = 3;

std::vector<polygon_t> polygons;

int frames_number;
std::string output_path;
int frame_w, frame_h;
double view_angle;
camera_t camera;
std::vector<figure_t> figures(FIGURES_NUMBER);
floor_t scene_floor;
int light_sources_number;
std::vector<light_source_t> light_sources;
int recursion_depth, ssaa_coef;

void readData() {
    std::cin >> frames_number;
    std::cin >> output_path;
    std::cin >> frame_w >> frame_h >> view_angle;
    std::cin >> camera;
    for (int i = 0; i < FIGURES_NUMBER; ++i) {
        std::cin >> figures[i];
    }
    std::cin >> scene_floor;
    std::cin >> light_sources_number;
    light_sources.resize(light_sources_number);
    for (int i = 0; i < light_sources_number; ++i) {
        std::cin >> light_sources[i];
    }
    std::cin >> recursion_depth >> ssaa_coef;
}

void printDefault() {
    FILE *file = fopen("default.in", "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    int c;
    while ((c = getc(file)) != EOF) {
        putchar(c);
    }

    fclose(file);
}

void printHelp(char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
    << "Options:\n"
    << "  --help     Display this help message\n"
    << "  --cpu      Use CPU\n"
    << "  --gpu      Use GPU (default)\n"
    << "  --default  Display default parameters\n";
}


void makeFrames(bool is_used_gpu) {
    readData();
    buildScene(is_used_gpu, scene_floor, polygons, figures);

    int ssaa_w = frame_w * ssaa_coef;
    int ssaa_h = frame_h * ssaa_coef;

    uchar4 *data = new uchar4[frame_w * frame_h];
    uchar4 *ssaa_data;
    uchar4 *ssaa_data_dev;
    uchar4 *data_dev;
    light_source_t* light_sources_dev;
    polygon_t* polygons_dev;
    if (is_used_gpu) {
        CSC(cudaMalloc(&light_sources_dev, sizeof(light_source_t) * light_sources_number));
        CSC(cudaMemcpy(light_sources_dev, light_sources.data(), sizeof(light_source_t) * light_sources_number, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&polygons_dev, sizeof(polygon_t) * polygons.size()));
        CSC(cudaMemcpy(polygons_dev, polygons.data(), sizeof(polygon_t) * polygons.size(), cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&ssaa_data_dev, sizeof(uchar4) * ssaa_w * ssaa_h));
        CSC(cudaMalloc(&data_dev, sizeof(uchar4) * frame_w * frame_h));
    } else {
        ssaa_data = new uchar4[ssaa_w * ssaa_h];
    }

    char output_filename[128];
    double tau = 2 * M_PI / frames_number;
    for (int frame = 0; frame < frames_number; ++frame) {
        double t = frame * tau;
        vec3d camera_pc = vec3d::fromCylindrical(camera.r0_c + camera.Ar_c * sin(camera.wr_c * t + camera.pr_c), camera.z0_c + camera.Az_c * sin(camera.wz_c * t + camera.pz_c), camera.phi0_c + camera.wphi_c * t);
        vec3d camera_pv = vec3d::fromCylindrical(camera.r0_n + camera.Ar_n * sin(camera.wr_n * t + camera.pr_n), camera.z0_n + camera.Az_n * sin(camera.wz_n * t + camera.pz_n), camera.phi0_n + camera.wphi_n * t);
        if (is_used_gpu) {
            gpuRender(frame, camera_pc, camera_pv, ssaa_w, ssaa_h, view_angle, ssaa_data_dev, light_sources_dev, polygons_dev, light_sources_number, polygons.size(), recursion_depth);
            gpuSsaa<<<BLOCKS_2D, THREADS_2D>>>(ssaa_data_dev, data_dev, frame_w, frame_h, ssaa_coef);
            CSC(cudaMemcpy(data, data_dev, sizeof(uchar4) * frame_w * frame_h, cudaMemcpyDeviceToHost));
        } else {
            cpuRender(frame, camera_pc, camera_pv, ssaa_w, ssaa_h, view_angle, ssaa_data, polygons, light_sources, recursion_depth);
            cpuSsaa(ssaa_data, data, frame_w, frame_h, ssaa_coef);
        }
        sprintf(output_filename, output_path.c_str(), frame);
        writeImageData(output_filename, frame_w, frame_h, data);
    }
    free(data);
    if (is_used_gpu) {
        CSC(cudaFree(ssaa_data_dev));
        CSC(cudaFree(data_dev));
        CSC(cudaFree(light_sources_dev));
        CSC(cudaFree(polygons_dev));
        CSC(cudaGetLastError());
    } else {
        free(ssaa_data);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2 && argc != 1) {
        printHelp(argv[0]);
        return 1;
    }
    bool is_used_gpu = true;
    for (int i = 1; i < argc; ++i) {
        char* a = argv[i];
        if (strcmp(a, "--help") == 0) {
            printHelp(argv[0]);
            return 0;
        }
        else if (strcmp(a, "--cpu") == 0) {
            is_used_gpu = false;
        } else if (strcmp(a, "--gpu") == 0) {
            is_used_gpu = true;
        } else if (strcmp(a, "--default") == 0) {
            printDefault();
            return 0;
        } else {
            printf("Wrong key: %s\n", a);
            printHelp(argv[0]);
            return 1;
        }
    }
    makeFrames(is_used_gpu);
    return 0;
}
