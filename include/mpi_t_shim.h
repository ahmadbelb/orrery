#ifndef ORRERY_MPI_T_SHIM_H
#define ORRERY_MPI_T_SHIM_H
#include <mpi.h>

typedef void* MPI_T_cvar_handle;

static inline int MPI_T_init_thread(int required, int* provided) {
    if (provided)
        *provided = required;
    return MPI_SUCCESS;
}
static inline int MPI_T_cvar_get_index(const char* name, int* index) {
    (void)name;
    (void)index;
    return MPI_ERR_OTHER;
}
static inline int MPI_T_cvar_handle_alloc(int index, void* obj, MPI_T_cvar_handle* h, int* count) {
    (void)index;
    (void)obj;
    (void)count;
    if (h)
        *h = 0;
    return MPI_ERR_OTHER;
}
static inline int MPI_T_cvar_handle_free(MPI_T_cvar_handle* h) {
    if (h)
        *h = 0;
    return MPI_SUCCESS;
}
static inline int MPI_T_cvar_write(MPI_T_cvar_handle h, const void* val) {
    (void)h;
    (void)val;
    return MPI_SUCCESS;
}
static inline int MPI_T_finalize(void) {
    return MPI_SUCCESS;
}
#endif
