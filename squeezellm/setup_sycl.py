from setuptools import Extension, setup
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

setup(
    name="quant_cuda",
    ext_modules=[
        DPCPPExtension(
            "quant_cuda", ["quant_sycl.cpp", "quant_sycl_kernel.cpp"],
            include_dirs=ipex.xpu.cpp_extension.include_paths(),
        )
    ],
    cmdclass={"build_ext": DpcppBuildExtension},
)