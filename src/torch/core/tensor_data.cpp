/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor_data.h"

#include "torch/cuda/cached_memory_allocator.h"

#ifdef TT_HAS_CUDA
// #    include <sys/mman.h>

#    include "torch/cuda/ops_impl_cuda_helper.h"
#    include "torch/cuda/tt_cuda.h"
#    include <cuda_runtime.h>
#endif


/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#    include <windows.h>
//
#    include <psapi.h>



#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#    include "sys/sysinfo.h"
#    include "sys/types.h"

#    include <sys/resource.h>
#    include <unistd.h>

#    if defined(__APPLE__) && defined(__MACH__)
#        include <mach/mach.h>

#    elif (defined(_AIX) || defined(__TOS__AIX__)) || \
        (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#        include <fcntl.h>
#        include <procfs.h>

#    elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#        include <fstream>
#        include <stdio.h>

#    endif

#else

#endif


static size_t getMaxSystemMemory()
{
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
#else
    size_t pages     = sysconf(_SC_PHYS_PAGES);
    size_t page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
#endif
}


#if !defined(_WIN32)
inline long long get_val(const std::string& target, const std::string& content)
{
    long long result  = -1;
    std::size_t start = content.find(target);
    if (start != std::string::npos)
    {
        int begin          = start + target.length();
        std::size_t end    = content.find("kB", start);
        std::string substr = content.substr(begin, end - begin);
        result             = std::stoll(substr) * 1000;
    }
    return result;
}
#endif

static size_t getUsedSystemMemory()
{
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys - status.ullAvailPhys;
#else
    std::ifstream proc_meminfo("/proc/meminfo");
    if (proc_meminfo.good())
    {
        std::string content((std::istreambuf_iterator<char>(proc_meminfo)), std::istreambuf_iterator<char>());
        size_t total = get_val("MemTotal:", content);
        size_t avail = get_val("MemAvailable:", content);
        return total - avail;
    }

    struct sysinfo memInfo;
    sysinfo(&memInfo);

    size_t physMemUsed = memInfo.totalram - memInfo.freeram;
    // Multiply in next statement to avoid int overflow on right hand side...
    physMemUsed *= memInfo.mem_unit;
    return physMemUsed;

#endif
}
/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || \
    (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1) return (size_t)0L; /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L; /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#    if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#    else
    return (size_t)(rusage.ru_maxrss * 1024L);
#    endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L; /* Unsupported. */
#endif
}



/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L; /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    int64_t rss = 0L;
    FILE* fp    = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL) return (size_t)0L; /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1)
    {
        fclose(fp);
        return (size_t)0L; /* Can't read? */
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L; /* Unsupported. */
#endif
}

static void print_memory_info(std::ostream& strm)
{
    auto current_memory_used  = getCurrentRSS();
    auto max_memory_used      = getPeakRSS();
    auto max_memory_available = getMaxSystemMemory();
    auto total_memory_used    = getUsedSystemMemory();

    strm << "[Memory Info]\n";
    strm << "Current Usage (MB): " << current_memory_used / (1000.0 * 1000.0) << "\n";
    strm << "Max Usage (MB):     " << max_memory_used / (1000.0 * 1000.0) << "\n";
    strm << "Max Available (MB): " << max_memory_available / (1000.0 * 1000.0)<< "\n";
    strm << "Total used (MB): " << total_memory_used / (1000.0 * 1000.0) << std::endl;
}



static void* malloc_impl_host(int64_t size)
{
    auto ptr = malloc(size);
    // auto ptr = calloc(size, 1);
    //    auto ptr = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    // memset(ptr,0,size);
    return ptr;
}

static void free_impl_host(void* ptr, int64_t size)
{
    free(ptr);
    //    munmap(ptr, size);
}


namespace tinytorch
{
StorageImpl::StorageImpl(int64_t size, TensorOptions __options) : size_(size), options_(__options)
{
    if (options_.device_ == kCPU)
    {
#ifdef TT_HAS_CUDA
        if (options_.pinned_memory_)
        {
            data_ptr_ = cuda::cuda_malloc_pinned(size);
            if (!data_ptr_)
            {
                options_.pinned_memory_ = false;
            }
        }

        if (!options_.pinned_memory_)
        {
            CHECK(!data_ptr_);
            data_ptr_ = malloc_impl_host(size);
        }
#else
        data_ptr_ = malloc_impl_host(size);
#endif

#if TT_DEBUG
        memset(data_ptr_, 0xabababab, size);
#endif
        has_ownership = true;

        if (!data_ptr_)
        {
            print_memory_info(std::cout);
            throw TinyTorchException(std::string("CPU memory allocation failed. Out of memory."),
                                     TinyTorchExceptionStatus::OutOfMemoryCPU);
        }
    }
    else if (options_.device_ == kCUDA)
    {
#ifdef TT_HAS_CUDA
        cuda::DeviceGuard g(options_.device_);

        std::tie(data_ptr_, alloc_info) = cuda::cuda_cached_malloc(size, options_.device_.index());
#    if TT_DEBUG
        TT_CHECK_CUDA_ERROR(cudaMemsetAsync(data_ptr_, 0xabababab, size, cuda::getCurrentCUDAStream()));
#    endif

        has_ownership = true;
#else
        CHECK(false);
#endif
    }
    else
    {
        CHECK(false) << "invalid device type " << options_.device_;
    }
}


StorageImpl::StorageImpl(void* data_ptr, int64_t size, uint64_t alloc_info, TensorOptions options)
    : size_(size), alloc_info(alloc_info), options_(options)
{
    data_ptr_     = data_ptr;
    has_ownership = false;
}


StorageImpl::~StorageImpl()
{
    if (has_ownership == true)
    {
        if (options_.device_ == kCPU)
        {
#ifdef TT_HAS_CUDA
            if (options_.pinned_memory_)
            {
                cuda::cuda_pinned_free(data_ptr_, size_);
            }
            else
            {
                free_impl_host(data_ptr_, size_);
            }
#else
            free_impl_host(data_ptr_, size_);
#endif
        }
        else
        {
#ifdef TT_HAS_CUDA
            cuda::DeviceGuard g(options_.device_);

            cuda::cuda_cached_free(data_ptr_, alloc_info, options_.device_.index());
#else
            CHECK(false);
#endif
        }
    }
}

}  // namespace tinytorch