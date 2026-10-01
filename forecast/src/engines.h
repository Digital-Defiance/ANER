#ifndef FORECAST_ENGINES_H
#define FORECAST_ENGINES_H

/* Stationary chain: Y = A^depth X.
   A is channels × channels, X and Y are channels × spatial, row-major. */
int cpu_chain(const float *A, int channels, int depth, const float *X, float *Y, int spatial);

/* On success writes milliseconds of one execution into *ms when ms != NULL. */
int gpu_chain(const float *A, int channels, int depth, const float *X, float *Y, int spatial, double *ms);

/* compile_ms is nonzero only on the first use of a (channels, spatial, depth, A) program. */
int ane_chain(const float *A, int channels, int depth, const float *X, float *Y, int spatial,
              double *compile_ms, double *exec_ms);
void ane_shutdown(void);

#endif
