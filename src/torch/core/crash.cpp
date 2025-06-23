/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "crash.h"


#include <iostream>
#include <sstream>


#if defined(_WIN32)
#    include <signal.h>
#    include <stdio.h>
#    include <stdlib.h>
#    include <tchar.h>
#    include <windows.h>
// dbghelp must be included after windows.h
#    include <DbgHelp.h>
#endif


#if defined(__unix__)
#    include <execinfo.h>
#    include <signal.h>
#    include <stdio.h>
#    include <stdlib.h>
#    include <string.h>
#    include <ucontext.h>
#    include <unistd.h>
#endif

namespace tinytorch
{
#if defined(__unix__)
// Source: http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes

/* This structure mirrors the one found in /usr/include/asm/ucontext.h */
typedef struct _sig_ucontext
{
    uint64_t uc_flags;
    struct ucontext* uc_link;
    stack_t uc_stack;
    struct sigcontext uc_mcontext;
    sigset_t uc_sigmask;
} sig_ucontext_t;


std::string printCurrentStack()
{
    std::stringstream  strm;
    strm << "printCurrentStack... " << std::endl;
    void* array[50];
    char** messages;
    int size, i;

    size = backtrace(array, 50);

    /* overwrite sigaction with caller's address */
    //    array[1] = caller_address;

    messages = backtrace_symbols(array, size);



    /* skip first stack frame (points here) */
    for (i = 0; i < size && messages != NULL; ++i)
    {
        strm<< "[bt]: (" << i << ") " << messages[i] << std::endl;
    }

    free(messages);

    return strm.str();
}

#endif

#if defined(_WIN32)
// The code requires you to link against the DbgHelp.lib library

std::string printCurrentStack()
{
    std::stringstream  strm;
    std::string outWalk;
    // Set up the symbol options so that we can gather information from the current
    // executable's PDB files, as well as the Microsoft symbol servers.  We also want
    // to undecorate the symbol names we're returned.  If you want, you can add other
    // symbol servers or paths via a semi-colon separated list in SymInitialized.
    ::SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_INCLUDE_32BIT_MODULES | SYMOPT_UNDNAME);
    if (!::SymInitialize(::GetCurrentProcess(), "http://msdl.microsoft.com/download/symbols", TRUE)) return "";

    // Capture up to 25 stack frames from the current call stack.  We're going to
    // skip the first stack frame returned because that's the GetStackWalk function
    // itself, which we don't care about.
    const int numAddrs    = 50;
    PVOID addrs[numAddrs] = {0};
    USHORT frames         = CaptureStackBackTrace(0, numAddrs - 1, addrs, NULL);

    for (USHORT i = 0; i < frames; i++)
    {
        // Allocate a buffer large enough to hold the symbol information on the stack and get
        // a pointer to the buffer.  We also have to set the size of the symbol structure itself
        // and the number of bytes reserved for the name.
        ULONG64 buffer[(sizeof(SYMBOL_INFO) + 1024 + sizeof(ULONG64) - 1) / sizeof(ULONG64)] = {0};
        SYMBOL_INFO* info                                                                    = (SYMBOL_INFO*)buffer;
        info->SizeOfStruct                                                                   = sizeof(SYMBOL_INFO);
        info->MaxNameLen                                                                     = 1024;

        // Attempt to get information about the symbol and add it to our output parameter.
        DWORD64 displacement = 0;
        if (::SymFromAddr(::GetCurrentProcess(), (DWORD64)addrs[i], &displacement, info))
        {
            // outWalk.append(info->Name, info->NameLen);
            // outWalk.append("\n");
            strm << "[bt]: (" << i << ") " << info->Name << std::endl;
        }
    }

    ::SymCleanup(::GetCurrentProcess());
    return strm.str();
}


#endif



}  // namespace Saiga
