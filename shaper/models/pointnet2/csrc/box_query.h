#ifndef _BOX_QUERY
#define _BOX_QUERY

#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> BoxQuery(
    const at::Tensor points,
    const at::Tensor centroids,
    const at::Tensor lens,
    const int64_t num_neighbours);

#endif
