#ifndef _KNN_QUERY
#define _KNN_QUERY

#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> knn(
    at::Tensor & query,
    at::Tensor & key,
    const int k
    );

#endif

