#ifndef __GOTORCH_API_H__
#define __GOTORCH_API_H__

#ifdef __cplusplus
extern "C"
{
	typedef torch::Tensor* tensor;
	typedef torch::optim::Optimizer* optimizer;
#else
	typedef void* tensor;
	typedef void* optimizer;
#endif

#ifdef GOTORCH_EXPORT
#define GOTORCH_API __declspec(dllexport)
#else
#define GOTORCH_API
#endif

#ifdef __cplusplus
}
#endif

#endif // !__GOTORCH_API_H__
