add_executable(app app.cu)
target_compile_features(app PRIVATE cxx_std_17)

target_link_libraries(app PRIVATE
        raytracer
        Eigen3::Eigen
        fmt::fmt)
