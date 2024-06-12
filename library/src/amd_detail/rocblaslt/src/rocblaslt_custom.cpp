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

#include "rocblaslt-custom.h"

#include <Tensile/Utils.hpp>

#define FUSED_SILU_SOL_INDEX0 -1

rocblaslt_status
    rocBLASLtGemmCustomKernelFinder(rocblaslt_handle                       handle,
                                    const RocblasltCustomKernelType        custom,
                                    const Tensile::ContractionProblemGemm& problem,
                                    const Tensile::ContractionInputs&      inputs,
                                    const int                              workspaceBytes,
                                    const int                              requestedAlgoCount,
                                    std::vector<rocblaslt_matmul_heuristic_result>& results)
{
    rocblaslt_status status = rocblaslt_status_success;
    if(custom == RocblasltCustomKernelType::FUSE_SILU)
    {
        std::cout << "Hello" << std::endl;
        results.clear();
        if(!(problem.transA() && !problem.transB())
           || !(problem.a().dataType() == Tensile::DataType::BFloat16 && problem.b().dataType() == Tensile::DataType::BFloat16)
           || problem.freeSizeA(0) != 7168 || !(problem.freeSizeB(0) >= 1 && problem.freeSizeB(0) <= 4)
           || problem.boundSize(0) != 8192)
        {
            std::cout << problem.transA() << " " << problem.transB() << " " << !(problem.transA() && !problem.transB()) << std::endl
                      << problem.a().dataType() << " " << problem.b().dataType() << " "
                      << problem.freeSizeA(0) << " " << problem.freeSizeB(0) << " "
                      << problem.boundSize(0) << std::endl;
            return status;
        }

        std::vector<int> solutions;
        solutions.push_back(FUSED_SILU_SOL_INDEX0); // solution index
        int i = 0;
        results.resize(solutions.size());
        for(auto solution : solutions)
        {
            // Fixed lines below
            memset(&results[i], 0, sizeof(rocblaslt_matmul_heuristic_result));
            memset(results[i].algo.data, 0, sizeof(results[i].algo.data));
            int* solutionIndex                           = (int*)(results[i].algo.data);
            results[i].algo.max_workspace_bytes = workspaceBytes;  // Fixed value
            results[i].algo.fallback            = false;    // Fixed value
            results[i].state                    = rocblaslt_status_success;  // Fixed value
            // Fixed line above

            *solutionIndex                    = solution;  // Set to a negative value
            results[i].workspaceSize = 0;  // Set to 0 if not needed
            i++;
        }
    }
    else
    {
        status = rocblaslt_status_not_implemented;
    }

    return status;
}

rocblaslt_status rocBLASLtIsKernelSupported(rocblaslt_handle                       handle,
                                            const int                              index,
                                            const Tensile::ContractionProblemGemm& problem,
                                            const Tensile::ContractionInputs&      inputs,
                                            size_t* workspaceSizeInBytes)
{
    rocblaslt_status status = rocblaslt_status_invalid_value;
    // Check if the kernel is supported with the given solution index.
    // Return the size of workspaceSizeInBytes if successful.
    if(index == FUSED_SILU_SOL_INDEX0)
    {
        if((problem.transA() && !problem.transB())
           || problem.a().dataType() == Tensile::DataType::BFloat16 || problem.b().dataType() == Tensile::DataType::BFloat16
           || problem.freeSizeA(0) != 7168 || (problem.freeSizeB(0) >= 1 && problem.freeSizeB(0) <= 4)
           || problem.boundSize(0) != 8192)
            return status;
        status = rocblaslt_status_success;
    }
    return status;
}

template <bool T_Debug>
Tensile::KernelInvocation outputInvocationTemplate(const Tensile::ContractionProblemGemm& problem,
                                                   const Tensile::ContractionInputs&      inputs,
                                                   void*                                  workspace,
                                                   hipStream_t                            stream)
{
    Tensile::KernelInvocation rv;
    rv.args = Tensile::KernelArguments(T_Debug);
    rv.args.reserve(512, 64);
    rv.kernelName = "sampleKernel"; // Write your hip kernel function name here
    rv.codeObjectFile = "hipblasltGemmSilu.hsaco"; // Write your hip code object file name here

    // Start modify here
    rv.workGroupSize.x = 256;
    rv.workGroupSize.y = 1;
    rv.workGroupSize.z = 1;

    rv.numWorkGroups.x = std::ceil(problem.problemSizes()[0] * problem.problemSizes()[1] / rv.workGroupSize.x);
    rv.numWorkGroups.y = 1;
    rv.numWorkGroups.z = 1;

    rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
    rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
    rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

    // This is how the kernel arguments are passed to the kernel
    // Put them in order with rv.args.append<type>(...)
    Tensile::TensorDescriptor const& a = problem.a();
    Tensile::TensorDescriptor const& b = problem.b();
    Tensile::TensorDescriptor const& c = problem.c();
    Tensile::TensorDescriptor const& d = problem.d();
    // How to get sizes, m, n, b, k
    int idx = 0;
    for(auto size : problem.problemSizes())
    {
        rv.args.append<uint32_t>(Tensile::concatenate_if<T_Debug>("size_", idx), size);
        idx++;
    }
    // Stride D, C, A, B, assuming strides[0] is 1, i starts from 1
    for(size_t i = 1; i < d.dimensions(); i++)
        rv.args.append<uint32_t>(Tensile::concatenate_if<T_Debug>("strideD", i), d.strides()[i]);

    for(size_t i = 1; i < c.dimensions(); i++)
        rv.args.append<uint32_t>(Tensile::concatenate_if<T_Debug>("strideC", i), c.strides()[i]);

    for(size_t i = 1; i < a.dimensions(); i++)
    {
        auto stride_a = a.strides()[i];
        rv.args.append<uint32_t>(Tensile::concatenate_if<T_Debug>("strideA", i), stride_a);
    }

    for(size_t i = 1; i < b.dimensions(); i++)
    {
        auto stride_b = b.strides()[i];
        rv.args.append<uint32_t>(Tensile::concatenate_if<T_Debug>("strideB", i), stride_b);
    }

    // How to get input pointers
    rv.args.append<void const*>("A", inputs.a);
    rv.args.append<void const*>("B", inputs.b);
    rv.args.append<void*>("D", inputs.d);

    // How to get constant variant if your alpha is float
    rv.args.append<float>("alpha", *std::get_if<float>(&inputs.alpha));
    return rv;
}

rocblaslt_status rocBLASLtMakeArgument(rocblaslt_handle                        handle,
                                       const int                               index,
                                       const Tensile::ContractionProblemGemm&  problem,
                                       const Tensile::ContractionInputs&       inputs,
                                       void*                                   workspace,
                                       hipStream_t                             stream,
                                       std::vector<Tensile::KernelInvocation>& kernelInvocations)
{
    rocblaslt_status                       status = rocblaslt_status_success;
    std::vector<Tensile::KernelInvocation> rv;
    // Make the argument for the custom kernel with the given solution index.
    // Return the kernelInvocations if successful.
    /*
        if(index == -1)
        {
            rv.push_back(outputInvocationTemplate1(problem, inputs, workspace, stream));
        }
        else if(index == -2)
        {
            rv.push_back(outputInvocationTemplate2(problem, inputs, workspace, stream));
        }
        else
        {
            return status;
        }
     */
    //

    rv.push_back(outputInvocationTemplate<false>(problem, inputs, workspace, stream));

    kernelInvocations = rv;
    return status;
}
