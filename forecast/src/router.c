#include "router.h"

#include <stdio.h>

Engine route_engine(int depth, int channels, int spatial, int requires_fp32) {
    if (requires_fp32)
        return ENGINE_CPU;
    if (spatial >= 2048)
        return ENGINE_GPU;
    if (depth == 1 && channels <= 1024)
        return ENGINE_CPU;
    if (depth >= 32 && channels <= 512 && spatial <= 64)
        return ENGINE_ANE;
    if (depth == 1 && spatial <= 64 && channels >= 2048)
        return ENGINE_ANE;
    return ENGINE_CPU;
}

int plan_rollout(int channels, int scenarios, int depth, int requires_fp32, Tile *tiles, int cap) {
    if (scenarios <= 0 || cap < 1)
        return -1;
    if (requires_fp32 || !(depth >= 32 && channels <= 512)) {
        tiles[0].engine = route_engine(depth, channels, scenarios, requires_fp32);
        tiles[0].scenario0 = 0;
        tiles[0].spatial = scenarios;
        return 1;
    }
    int n = 0;
    for (int s = 0; s < scenarios;) {
        int sp = scenarios - s;
        if (sp > 64)
            sp = 64;
        if (n >= cap)
            return -1;
        tiles[n].engine = route_engine(depth, channels, sp, 0);
        tiles[n].scenario0 = s;
        tiles[n].spatial = sp;
        n++;
        s += sp;
    }
    return n;
}

const char *engine_name(Engine engine) {
    switch (engine) {
    case ENGINE_CPU: return "CPU";
    case ENGINE_GPU: return "GPU";
    case ENGINE_ANE: return "Neural Engine";
    }
    return "?";
}

int router_audit(void) {
    struct {
        int depth, channels, spatial, fp32;
        Engine winner;
    } cases[] = {
        {1, 512, 64, 0, ENGINE_CPU},
        {1, 1024, 64, 0, ENGINE_CPU},
        {1, 2048, 64, 0, ENGINE_ANE},
        {32, 512, 64, 0, ENGINE_ANE},
        {64, 512, 64, 0, ENGINE_ANE},
        {128, 512, 64, 0, ENGINE_ANE},
        {256, 256, 64, 0, ENGINE_ANE},
        {16, 512, 2048, 0, ENGINE_GPU},
        {4, 2048, 2048, 0, ENGINE_GPU},
        {1, 4096, 4096, 0, ENGINE_GPU},
        {128, 512, 64, 1, ENGINE_CPU},
    };
    int misses = 0;
    int n = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int i = 0; i < n; i++) {
        Engine got = route_engine(cases[i].depth, cases[i].channels, cases[i].spatial, cases[i].fp32);
        if (got != cases[i].winner)
            misses++;
    }
    Tile tiles[128];
    int nt = plan_rollout(512, 4096, 128, 0, tiles, 128);
    if (nt != 64)
        misses++;
    for (int i = 0; i < nt && misses == 0; i++) {
        if (tiles[i].engine != ENGINE_ANE || tiles[i].spatial != 64)
            misses++;
    }
    if (route_engine(1, 512, 4096, 0) != ENGINE_GPU)
        misses++;
    return misses;
}
