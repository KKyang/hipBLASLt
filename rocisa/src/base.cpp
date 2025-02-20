/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

#include "hardware_caps.hpp"
#include "helper.hpp"

namespace nb = nanobind;

namespace base
{
    struct KernelInfo
    {
        IsaVersion isaVersion;
        int        wavefront;
    };

    struct IsaInfo
    {
        std::map<std::string, int>  asm_caps;
        std::map<std::string, bool> arch_caps;
        std::map<std::string, int>  reg_caps;
        std::map<std::string, bool> asm_bugs;
    };

    class rocIsa
    {
    public:
        // Delete copy constructor and assignment operator
        rocIsa(const rocIsa&)            = delete;
        rocIsa& operator=(const rocIsa&) = delete;

        // Static method to get the single instance of the class
        static rocIsa& getInstance()
        {
            std::call_once(initInstanceFlag, &rocIsa::init);
            return *instance;
        }

        void initIsa(const nb::tuple& arch, const std::string& assemblerPath, bool debug = false)
        {
            IsaVersion isaVersion
                = {nb::cast<int>(arch[0]), nb::cast<int>(arch[1]), nb::cast<int>(arch[2])};
            if(m_isainfo.find(isaVersion) != m_isainfo.end())
                return;
            // Init ISA
            IsaInfo isainfo;
            isainfo.asm_caps      = initAsmCaps(isaVersion, assemblerPath, debug);
            isainfo.arch_caps     = initArchCaps(isaVersion);
            isainfo.reg_caps      = initRegisterCaps(isaVersion, isainfo.arch_caps);
            isainfo.asm_bugs      = initAsmBugs(isainfo.asm_caps);
            m_isainfo[isaVersion] = isainfo;
        }

        bool isInitIsa()
        {
            return (m_isainfo.size() > 0);
        }

        void setKernel(const nb::tuple& arch, const int wavefrontSize)
        {
            std::thread::id id = std::this_thread::get_id();
            IsaVersion      isaVersion
                = {nb::cast<int>(arch[0]), nb::cast<int>(arch[1]), nb::cast<int>(arch[2])};
            m_mutex.lock();
            m_threads[id] = {isaVersion, wavefrontSize};
            m_mutex.unlock();
        }

        KernelInfo getKernel()
        {
            return m_threads[std::this_thread::get_id()];
        }

        std::map<std::string, int> getAsmCaps()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].asm_caps;
        }

        std::map<std::string, int> getRegCaps()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].reg_caps;
        }

        std::map<std::string, bool> getArchCaps()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].arch_caps;
        }

        std::map<std::string, bool> getAsmBugs()
        {
            return m_isainfo[m_threads[std::this_thread::get_id()].isaVersion].asm_bugs;
        }

    private:
        rocIsa() {}

        static void init()
        {
            instance = new rocIsa();
        }

        std::mutex                            m_mutex;
        std::map<std::thread::id, KernelInfo> m_threads;
        std::map<IsaVersion, IsaInfo>         m_isainfo;

        static rocIsa*        instance;
        static std::once_flag initInstanceFlag;
    };

    // Initialize static members
    rocIsa*        rocIsa::instance = nullptr;
    std::once_flag rocIsa::initInstanceFlag;

    const char* isaToGfx(const nb::tuple& arch)
    {
        /*Converts an ISA version to a gfx architecture name.

        Args:
            arch: An object representing the major, minor, and step version of the ISA.
    
        Returns:
            The name of the GPU architecture (e.g., 'gfx906').
        */
        return (std::string("gfx") + std::to_string(nb::cast<int>(arch[0]))
                + std::to_string(nb::cast<int>(arch[1]))
                + std::to_string(nb::cast<int>(arch[2]))).c_str();
    }
}

NB_MODULE(rocisa, m)
{
    m.doc() = "Module rocisa.";
    m.def("isaToGfx", &base::isaToGfx);

    nb::class_<base::rocIsa>(m, "rocIsa")
        .def_static("getInstance", &base::rocIsa::getInstance, nb::rv_policy::reference)
        .def("init", &base::rocIsa::initIsa, "Init ISA.")
        .def("isInit", &base::rocIsa::isInitIsa, "Check if ISA is init.")
        .def("setKernel", &base::rocIsa::setKernel, "Set kernel.")
        .def("getKernel", &base::rocIsa::getKernel, "Get kernel.")
        .def("getAsmCaps", &base::rocIsa::getAsmCaps, "Get asm caps.")
        .def("getRegCaps", &base::rocIsa::getRegCaps, "Get reg caps.")
        .def("getArchCaps", &base::rocIsa::getArchCaps, "Get arch caps.")
        .def("getAsmBugs", &base::rocIsa::getAsmBugs, "Get asm bugs.");

    nb::class_<IsaVersion>(m, "IsaVersion")
        .def(nb::init<>())
        .def("__getitem__",
             [](const IsaVersion& a, size_t i) {
                 if(i >= a.size())
                     throw std::out_of_range("Index out of range");
                 return a[i];
             })
        .def("__setitem__",
             [](IsaVersion& a, size_t i, int v) {
                 if(i >= a.size())
                     throw std::out_of_range("Index out of range");
                 a[i] = v;
             })
        .def("__len__", [](const IsaVersion& a) { return a.size(); })
        .def("__eq__",
             [](const IsaVersion& a, const IsaVersion& b) {
                 return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]);
             })
        .def("__str__", [](const IsaVersion& a) {
            std::string s;
            for(auto a_u : a)
                s += std::to_string(a_u);
            return s;
        })
        .def("__getstate__", [](const IsaVersion& self) { return nb::make_tuple(self[0], self[1], self[2]);})
        .def("__setstate__", [](IsaVersion& self, nb::tuple state)
         {
            self[0] = nb::cast<int>(state[0]);
            self[1] = nb::cast<int>(state[1]);
            self[2] = nb::cast<int>(state[2]);
        });

    nb::class_<base::KernelInfo>(m, "KernelInfo")
        .def(nb::init<>())
        .def_rw("isa", &base::KernelInfo::isaVersion)
        .def_rw("wavefrontSize", &base::KernelInfo::wavefront)
        .def("__getstate__", [](const base::KernelInfo& self) { return nb::make_tuple(self.isaVersion, self.wavefront);})
        .def("__setstate__", [](base::KernelInfo& self, nb::tuple state)
         {
            self.isaVersion = nb::cast<IsaVersion>(state[0]);
            self.wavefront  = nb::cast<int>(state[1]);
         });
}
