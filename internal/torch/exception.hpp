#ifndef __GOTORCH_EXCEPTION_H__
#define __GOTORCH_EXCEPTION_H__

#include "api.h"

#ifdef __cplusplus
template <typename Function>
tensor auto_catch_tensor(Function f, char **err)
{
    try
    {
        return f();
    }
    catch (const torch::Error &e)
    {
        *err = strdup(e.msg().c_str());
    }
    catch (const std::exception &e)
    {
        *err = strdup(e.what());
    }
    return nullptr;
}

template <typename Function>
void auto_catch_void(Function f, char **err)
{
    try
    {
        f();
    }
    catch (const torch::Error &e)
    {
        *err = strdup(e.msg().c_str());
    }
    catch (const std::exception &e)
    {
        *err = strdup(e.what());
    }
}

template <typename Function>
optimizer auto_catch_optimizer(Function f, char **err)
{
    try
    {
        return f();
    }
    catch (const torch::Error &e)
    {
        *err = strdup(e.msg().c_str());
    }
    catch (const std::exception &e)
    {
        *err = strdup(e.what());
    }
    return 0;
}

template <typename Function>
double auto_catch_double(Function f, char **err)
{
    try
    {
        return f();
    }
    catch (const torch::Error &e)
    {
        *err = strdup(e.msg().c_str());
    }
    catch (const std::exception &e)
    {
        *err = strdup(e.what());
    }
    return 0;
}

template <typename Function>
module auto_catch_module(Function f, char **err)
{
    try
    {
        return f();
    }
    catch (const torch::Error &e)
    {
        *err = strdup(e.msg().c_str());
    }
    catch (const std::exception &e)
    {
        *err = strdup(e.what());
    }
    return nullptr;
}

template <typename Function>
size_t auto_catch_size_t(Function f, char **err)
{
    try
    {
        return f();
    }
    catch (const torch::Error &e)
    {
        *err = strdup(e.msg().c_str());
    }
    catch (const std::exception &e)
    {
        *err = strdup(e.what());
    }
    return 0;
}

template <typename Function>
optimizer_state auto_catch_optimizer_state(Function f, char **err)
{
    try
    {
        return f();
    }
    catch (const torch::Error &e)
    {
        *err = strdup(e.msg().c_str());
    }
    catch (const std::exception &e)
    {
        *err = strdup(e.what());
    }
    return 0;
}

#endif

#endif // __GOTORCH_EXCEPTION_H__