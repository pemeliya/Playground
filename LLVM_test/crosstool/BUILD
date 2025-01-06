
package(default_visibility = ["//visibility:public"])

toolchain {
  toolchain_identifier: "rocm-toolchain-clang"
  host_system_name: "local"
  target_system_name: "local"
  target_cpu: "x86_64"
  target_libc: "local"
  compiler: "gcc"
  abi_version: "local"
  abi_libc_version: "local"

  builtin_sysroot: "/"

  # Add your custom include paths here
  cxx_builtin_include_directory: "/opt/rocm/include"

  tool_path { name: "gcc"; path: "/usr/bin/gcc" }
  tool_path { name: "g++"; path: "/usr/bin/g++" }
  tool_path { name: "ar"; path: "/usr/bin/ar" }
  tool_path { name: "ld"; path: "/usr/bin/ld" }
  tool_path { name: "nm"; path: "/usr/bin/nm" }
  tool_path { name: "objcopy"; path: "/usr/bin/objcopy" }
  tool_path { name: "objdump"; path: "/usr/bin/objdump" }
  tool_path { name: "strip"; path: "/usr/bin/strip" }

  cxx_flag: "-std=c++17"
  cxx_flag: "-isystem /opt/rocm/include"
  linker_flag: "-L/opt/rocm/lib"
}

# cc_toolchain_suite(
#     name = "custom_toolchain_suite",
#     toolchains = {
#         "k8": ":custom_toolchain",
#     },
# )

# cc_toolchain(
#     name = "custom_toolchain",
#     all_files = ":all_files",
#     compiler_files = ":compiler_files",
#     linker_files = ":linker_files",
#     objcopy_files = ":objcopy_files",
#     strip_files = ":strip_files",
#     toolchain_identifier = "custom_toolchain",
# )

toolchain(
    name = "toolchain-linux-x86_64",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    toolchain = ":cc-compiler-local",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain_suite(
    name = "rocm-toolchain-suite",
    toolchains = {
        "local|compiler": ":cc-compiler-local",
        "arm": ":cc-compiler-local",
        "aarch64": ":cc-compiler-local",
        "k8": ":cc-compiler-local",
        "piii": ":cc-compiler-local",
        "ppc": ":cc-compiler-local",
    },
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":all_files",
    compiler_files = ":compile_files",
    ar_files = ":ar_files",
    as_files = ":as_files",
    dwp_files = ":empty",
    linker_files = ":linker_files",
    objcopy_files = ":empty",
    strip_files = ":empty",
    # To support linker flags that need to go to the start of command line
    # we need the toolchain to support parameter files. Parameter files are
    # last on the command line and contain all shared libraries to link, so all
    # regular options will be left of them.
    supports_param_files = 1,
    toolchain_identifier = "rocm-toolchain-clang",
    # toolchain_config = ":cc-compiler-local-config",
)

filegroup(
    name = "all_files",
    srcs = glob([
        "/opt/rocm/lib/llvm/bin/*",                    # Include all required tool binaries
    ]),
)

filegroup(
    name = "compiler_files",
    srcs = ["/opt/rocm/lib/llvm/bin/clang", 
            "/opt/rocm/lib/llvm/bin/clang++"],
)

filegroup(
    name = "linker_files",
    srcs = ["/opt/rocm/lib/llvm/bin/lld-link"],
)

filegroup(
    name = "linker_files",
    srcs = ["/opt/rocm/lib/llvm/bin/lld-link"],
)

filegroup(
    name = "ar_files",
    srcs = ["/opt/rocm/lib/llvm/bin/llvm-ar"],
)

filegroup(
    name = "as_files",
    srcs = ["/opt/rocm/lib/llvm/bin/llvm-as"],
)

filegroup(
    name = "strip_files",
    srcs = ["/opt/rocm/lib/llvm/bin/llvm-strip"],
)

filegroup(
    name = "dwp_files",
    srcs = [],  # Add debug-related tools here if necessary
)

# cc_toolchain_config(
#     name = "cc-compiler-local-config",
#     cpu = "local",
#     compiler = "compiler",
#     toolchain_identifier = "local_linux",
#     host_system_name = "local",
#     target_system_name = "local",
#     target_libc = "local",
#     abi_version = "local",
#     abi_libc_version = "local",
#     cxx_builtin_include_directories = ["/opt/rocm/include"],
#     host_compiler_path = "/opt/rocm/lib/llvm/bin/clang++",
#     host_compiler_prefix = "/opt/rocm/",
#     compile_flags = [
#         "-U_FORTIFY_SOURCE",
#         "-fstack-protector",
#         "-Wall",
#         "-Wunused-but-set-parameter",
#         "-Wno-free-nonheap-object",
#         "-fno-omit-frame-pointer",
#         "-xrocm"
#     ],
#     opt_compile_flags = [
#         "-g0",
#         "-O2",
#         "-D_FORTIFY_SOURCE=1",
#         "-DNDEBUG",
#         "-ffunction-sections",
#         "-fdata-sections",
#     ],
#     dbg_compile_flags = ["-g"],
#     cxx_flags = ["-std=c++17"],
#     link_flags = [
#         "-fuse-ld=gold",
#         "-Wl,-no-as-needed",
#         "-Wl,-z,relro,-z,now",
#         "-pass-exit-codes",
#         "-lstdc++",
#         "-lm",
#     ],
#     link_libs = [],
#     opt_link_flags = [],
#     unfiltered_compile_flags = [
#         "-fno-canonical-system-headers",
#         "-Wno-builtin-macro-redefined",
#         "-D__DATE__=\"redacted\"",
#         "-D__TIMESTAMP__=\"redacted\"",
#         "-D__TIME__=\"redacted\"",
#     ], #+ [%{unfiltered_compile_flags}],
#     linker_bin_path = "/opt/rocm/hcc/compiler/bin",
#     coverage_compile_flags = ["--coverage"],
#     coverage_link_flags = ["--coverage"],
#     supports_start_end_lib = True,
# )
