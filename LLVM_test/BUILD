
cc_library(
    name = "hip_headers",
    includes = [
        "/opt/rocm-6.3.0.1/include",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "llvm_test",
    srcs = ["llvm_test.cc"],
    copts = [],
    deps = [
        "@rocm-repo//:hip",
        "@gpu-common//:includes",
        # Other dependencies here
    ],
)
