cmake_minimum_required(VERSION 3.5)
project(vb)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_C_FLAGS "-Wall -Werror -g -fsanitize=undefined -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CFLAGS} -fno-rtti -fno-exceptions")
else()
    set(CMAKE_C_FLAGS "-Wall -Werror -O2 -march=native -flto -fno-delete-null-pointer-checks -fno-strict-overflow -fno-strict-aliasing -ftrivial-auto-var-init=zero -D_FORTIFY_SOURCE=2 -finline-functions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CFLAGS} -fno-rtti -fno-exceptions")
endif()

option(VB_SAMPLE "Build samples" OFF)
option(VB_BUILD_DEPS "Build dependencies" ON)
find_package(SDL3 REQUIRED)

if (WIN32)
   set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
endif()

set(VB_INCLUDE_DIRS
    vb/
    ext/
    ext/volk/
    ext/VulkanMemoryAllocator/include/
)

set(VB_LIBS
    volk
    SDL3::SDL3
    VMA
    STB
    m
)

include_directories(
    ${VB_INCLUDE_DIRS}
    ext/imgui/
    ext/imgui/backends/
    ext/fastgltf/include/
)

add_library(${PROJECT_NAME}
    vb/vb.cc
)
add_library(vb::vb ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC 
    ${VB_INCLUDE_DIRS}
)

if(VB_BUILD_DEPS)
    add_subdirectory(ext/volk)
    add_library(VMA ext/vma.cc)
    add_library(STB ext/stb.cc)
endif()

if(SAMPLE)
    find_package(glm REQUIRED)
    add_compile_definitions(IMGUI_IMPL_VULKAN_USE_VOLK)
    option(FASTGLTF_COMPILE_AS_CPP20 ON)
    add_subdirectory(ext/fastgltf)
    add_library(IMGUI
	ext/imgui/imgui.cpp
	ext/imgui/imgui_demo.cpp
	ext/imgui/imgui_draw.cpp
	ext/imgui/imgui_tables.cpp
	ext/imgui/imgui_widgets.cpp
	ext/imgui/backends/imgui_impl_sdl3.cpp
	ext/imgui/backends/imgui_impl_vulkan.cpp
    )
    set(WORKING_SAMPLES 
	samples/compute.cc
	samples/gltf_textures.cc
    )
    foreach(file ${WORKING_SAMPLES})
        get_filename_component(sample ${file} NAME_WLE)
        set(SAMPLE_BINARY ${sample})
        add_executable(${SAMPLE_BINARY} ${file})
        target_link_libraries(${SAMPLE_BINARY}
       	    ${PROJECT_NAME}
	    ${VB_LIBS}
	    IMGUI
	    fastgltf::fastgltf
	    glm::glm
        )
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            target_link_libraries(${SAMPLE_BINARY} -fsanitize=undefined)
        endif()
    endforeach()
endif()
