#ifndef FORECAST_ROUTER_H
#define FORECAST_ROUTER_H

typedef enum {
    ENGINE_CPU = 0,
    ENGINE_GPU = 1,
    ENGINE_ANE = 2
} Engine;

typedef struct {
    Engine engine;
    int scenario0;
    int spatial;
} Tile;

Engine route_engine(int depth, int channels, int spatial, int requires_fp32);
int plan_rollout(int channels, int scenarios, int depth, int requires_fp32, Tile *tiles, int cap);
int router_audit(void);
const char *engine_name(Engine engine);

#endif
