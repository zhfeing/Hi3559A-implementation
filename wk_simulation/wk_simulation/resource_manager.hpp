#ifndef __INC_RESOURDE_MANAGER_HPP__
#define __INC_RESOURDE_MANAGER_HPP__

#pragma once

#ifdef __ARM_ARCH
#include <mpi_sys.h>
#endif // __ARM_ARCH
#include <hi_type.h>
#include <string>
#include "auxiliary.h"


template <typename T>
class ResourceManager
{
public:
    ResourceManager(std::string description = "");
    ResourceManager(size_t, std::string description = "");
    ~ResourceManager();
    /* malloc from ram cached only effective while "__ARM_ARCH" is defined */
    void malloc(size_t, bool is_cache = false);
    void free();
#ifdef __ARM_ARCH
    /* flush cache to ddr */
    void arm_flush_cache();
#endif // __ARM_ARCH

    /* return raw pointer, do not reshape the memory! In arm arch, it is virtual address pointer */
    T *get_raw_ptr();

    /* physic and virtual address are identity when "__ARM_ARCH" is not defined */
    HI_U64 get_physic_address() const;
    HI_U64 get_virtual_address() const;
    HI_U8 *get_virtual_address_ptr();

    /* check if the pointer is null */
    bool is_null() const;

    /* after assigment, all members pf right hand side will be moved to lhs and rhs will be set to nullptr */
    ResourceManager &operator=(ResourceManager&);

    /* act like array but do not check if offset is within the zone */
    T &operator[](size_t offset);

    /* check if offset is within the zone */
    T &at(size_t offset);

    /* get size which is the number of T blocks */
    size_t size() const;
private:
    /* copy construction is forbidden */
    ResourceManager(const ResourceManager&);
    ResourceManager(const ResourceManager&&);
    /* update physic and virtual address */
    void update_address();
    /* clear all descriptions about _resource */
    void reset_description();

#ifdef __ARM_ARCH
    void arm_malloc(bool is_cache = false);
    void arm_free();
#else
    void windows_malloc();
    void windows_free();
#endif

private:
    T *_resource;
    size_t _size;
    std::string _description;
    HI_U64 _physic_address;
    HI_U8 *_virtual_address_ptr;
};


template <typename T>
ResourceManager<T>::ResourceManager(std::string description)
    :_description(description), _physic_address(0), _virtual_address_ptr(nullptr)
{
    _resource = nullptr;
    _size = 0;
}

template <typename T>
ResourceManager<T>::ResourceManager(size_t size, std::string description)
    :_description(description), _physic_address(0), 
    _virtual_address_ptr(nullptr), _size(size)
{
#ifdef __ARM_ARCH
    arm_malloc();
#else
    windows_malloc();
#endif
}

template <typename T>
ResourceManager<T>::~ResourceManager()
{
    free();
}

template <typename T>
void ResourceManager<T>::malloc(size_t size, bool is_cache)
{
    if (!is_null())
    {
        throw bad_alloc();
    }
    _size = size;
#ifdef __ARM_ARCH
    arm_malloc(is_cache);
#else
    windows_malloc();
#endif

}

template <typename T>
void ResourceManager<T>::free()
{
#ifdef __ARM_ARCH
    arm_free();
#else
    windows_free();
#endif
}

template <typename T>
T *ResourceManager<T>::get_raw_ptr()
{
    return _resource;
}

template <typename T>
unsigned char *ResourceManager<T>::get_virtual_address_ptr()
{
    return _virtual_address_ptr;
}

template <typename T>
HI_U64 ResourceManager<T>::get_physic_address() const
{
    return _physic_address;
}

template <typename T>
HI_U64 ResourceManager<T>::get_virtual_address() const
{
    return reinterpret_cast<HI_U64>(_virtual_address_ptr);
}

template <typename T>
void ResourceManager<T>::reset_description()
{
    _size = 0;
    _resource = nullptr;
    _description.clear();
    _physic_address = 0;
    _virtual_address_ptr = nullptr;
}

template <typename T>
bool ResourceManager<T>::is_null() const
{
    return nullptr == _resource;
}

template <typename T>
ResourceManager<T> &ResourceManager<T>::operator=(ResourceManager& rhs)
{
    if (rhs != this)
    {
#ifdef __ARM_ARCH
        arm_free();
        _size = rhs._size;
        _resource = rhs._resource;
        _description = rhs._description;
        _physic_address = rhs._physic_address;
        _virtual_address_ptr = rhs._virtual_address_ptr;
        rhs.reset_description();
#else
        windows_free();
        _size = rhs._size;
        _resource = rhs._resource;
        _description = rhs._description;
        _physic_address = rhs._physic_address;
        _virtual_address_ptr = rhs._virtual_address_ptr;
        rhs.reset_description();
#endif // __ARM_ARCH
    }
}

template <typename T>
T &ResourceManager<T>::operator[](size_t offset)
{
    return _resource[offset];
}

template <typename T>
T &ResourceManager<T>::at(size_t offset)
{
    if (offset >= _size)
    {
        throw bad_out_of_range();
    }
    return _resource[offset];
}

template <typename T>
size_t ResourceManager<T>::size() const
{
    return _size;
}

#ifdef __ARM_ARCH
template <typename T>
void ResourceManager<T>::arm_malloc(bool is_cache)
{
    if (is_cache)
    {
        HI_S32 res = HI_MPI_SYS_MmzAlloc_Cached(&_physic_address, (void**)&_virtual_address_ptr,
            _description.c_str(), nullptr, _size * sizeof(T));
        if (HI_SUCCESS != res)
        {
            report_error("arm malloc cached", res);
        }
    }
    else
    {
        HI_S32 res = HI_MPI_SYS_MmzAlloc(&_physic_address, (void**)&_virtual_address_ptr,
            _description.c_str(), nullptr, _size * sizeof(T));
        if (HI_SUCCESS != res)
        {
            report_error("arm malloc", res);
        }
    }
    update_address();
}

template <typename T>
void ResourceManager<T>::arm_free()
{
    HI_S32 res = HI_MPI_SYS_MmzFree(_physic_address, _virtual_address_ptr);
    if(HI_SUCCESS != res)
    {
        report_error("HI_MPI_SYS_MmzFree", res);
    }
    reset_description();
}

template <typename T>
void ResourceManager<T>::arm_flush_cache()
{
    HI_S32 res = HI_MPI_SYS_MmzFlushCache(_physic_address, _virtual_address_ptr, _size * sizeof(T));
    if (HI_SUCCESS != res)
    {
        report_error("arm flush cache", res);
    }
    update_address();
}
#else
template <typename T>
void ResourceManager<T>::windows_malloc()
{
    _resource = new T[_size];
    update_address();
}

template <typename T>
void ResourceManager<T>::windows_free()
{
    if (nullptr != _resource)
    {
        delete[] _resource;
        reset_description();
    }
}
#endif

template <typename T>
void ResourceManager<T>::update_address()
{
#ifdef __ARM_ARCH
    _resource = reinterpret_cast<T *>(_virtual_address_ptr);
#else
    _virtual_address_ptr = reinterpret_cast<HI_U8 *>(_resource);
    _physic_address = reinterpret_cast<HI_U64>(_virtual_address_ptr);
#endif // __ARM_ARCH

}

#endif /* __INC_RESOURDE_MANAGER_HPP__ */