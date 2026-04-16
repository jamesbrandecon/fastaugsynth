using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using PackageCompiler

project_dir = normpath(joinpath(@__DIR__, ".."))
out_dir = joinpath(project_dir, "build", "dist")
mkpath(out_dir)

create_library(project_dir, out_dir;
    lib_name = "statlibbackend",
    incremental = false,
    force = true,
    filter_stdlibs = false,
    precompile_execution_file = joinpath(@__DIR__, "precompile_workload.jl"),
)

if Sys.isapple()
    lib_path = joinpath(out_dir, "lib", "libstatlibbackend.dylib")
    for rpath in ("@loader_path", "@loader_path/julia")
        run(`install_name_tool -add_rpath $rpath $lib_path`)
    end
end

println("Built backend library in: ", out_dir)
