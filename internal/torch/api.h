#ifndef __GOTORCH_API_H__
#define __GOTORCH_API_H__

#ifdef __cplusplus
#include <torch/torch.h>
extern "C"
{

    typedef torch::Tensor *tensor;
    typedef torch::optim::Optimizer *optimizer;
    typedef torch::nn::Module *module;
#else
typedef void *tensor;
typedef void *optimizer;
typedef void *module;
#endif

#if defined(GOTORCH_EXPORT) && defined(_WIN32)
#define GOTORCH_API __declspec(dllexport)
#else
#define GOTORCH_API
#endif

#ifdef __cplusplus
}
#endif

#endif // !__GOTORCH_API_H__
