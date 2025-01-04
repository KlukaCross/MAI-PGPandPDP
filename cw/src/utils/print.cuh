#pragma once

void print(int frame_id, float time, long long total_rays) {
    printf("%d\t%.3lf\t%lli\n", frame_id, time, total_rays);
}
