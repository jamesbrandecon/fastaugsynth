using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using PackageCompiler

project_dir = normpath(joinpath(@__DIR__, ".."))
out_dir = joinpath(project_dir, "build", "dist")
mkpath(out_dir)

create_library(project_dir, out_dir;
    lib_name = "statlibbackend",
    precompile_execution_file = nothing,
    incremental = false,
    force = true,
    filter_stdlibs = false,
)

println("Built backend library in: ", out_dir)
