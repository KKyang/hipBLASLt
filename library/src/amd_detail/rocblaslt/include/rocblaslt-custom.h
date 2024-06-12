/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief rocblaslt-custom.h provides custom functions connectors in rocblaslt
 */

#pragma once
#ifndef _ROCBLASLT_CUSTOM_H_
#define _ROCBLASLT_CUSTOM_H_

#include "rocblaslt-types.h"

#include <Tensile/Contractions.hpp>

enum RocblasltCustomKernelType
{
    FUSE_SILU = -1,
    NONE      = 0
};

rocblaslt_status
    rocBLASLtGemmCustomKernelFinder(rocblaslt_handle                       handle,
                                    const RocblasltCustomKernelType        custom,
                                    const Tensile::ContractionProblemGemm& problem,
                                    const Tensile::ContractionInputs&      inputs,
                                    const int                              workspaceBytes,
                                    const int                              requestedAlgoCount,
                                    std::vector<rocblaslt_matmul_heuristic_result>& results);

rocblaslt_status rocBLASLtIsKernelSupported(rocblaslt_handle                       handle,
                                            const int                              index,
                                            const Tensile::ContractionProblemGemm& problem,
                                            const Tensile::ContractionInputs&      inputs,
                                            size_t* workspaceSizeInBytes);

rocblaslt_status rocBLASLtMakeArgument(rocblaslt_handle                        handle,
                                       const int                               index,
                                       const Tensile::ContractionProblemGemm&  problem,
                                       const Tensile::ContractionInputs&       inputs,
                                       void*                                   workspace,
                                       hipStream_t                             stream,
                                       std::vector<Tensile::KernelInvocation>& kernelInvocations);
#endif
