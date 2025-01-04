#pragma once

struct camera_t {
    double r0_c, z0_c, phi0_c, Ar_c, Az_c, wr_c, wz_c, wphi_c, pr_c, pz_c;
    double r0_n, z0_n, phi0_n, Ar_n, Az_n, wr_n, wz_n, wphi_n, pr_n, pz_n;

    friend std::istream& operator>>(std::istream& in, camera_t& camera) {
        std::cin >> camera.r0_c >> camera.z0_c >> camera.phi0_c >> camera.Ar_c >> camera.Az_c >> camera.wr_c >> camera.wz_c >> camera.wphi_c >> camera.pr_c >> camera.pz_c;
        std::cin >> camera.r0_n >> camera.z0_n >> camera.phi0_n >> camera.Ar_n >> camera.Az_n >> camera.wr_n >> camera.wz_n >> camera.wphi_n >> camera.pr_n >> camera.pz_n;
        return in;
    }
};
