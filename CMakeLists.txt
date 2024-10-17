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

option(SAMPLE "Build samples" ON)

find_package(Vulkan REQUIRED)
find_package(SDL3 REQUIRED)
find_package(glm REQUIRED)

#set(FASTGLTF_COMPILE_AS_CPP20 ON)
#add_subdirectory(ext/fastgltf/)

include_directories(
    vb/
    ext/
    ext/VulkanMemoryAllocator/include/
    ext/imgui/
    ext/imgui/backends/
    #ext/fastgltf/include/
)

add_library(${PROJECT_NAME}
    vb/vb.cc
)

if(SAMPLE)
    add_library(VMA ext/vma.cc)
    add_library(STB ext/stb.cc)
    set(WORKING_SAMPLES 
	samples/imageless_framebuffer.cc
    )
    foreach(file ${WORKING_SAMPLES})
        get_filename_component(sample ${file} NAME_WLE)
        set(SAMPLE_BINARY ${sample})
        add_executable(${SAMPLE_BINARY} ${file})
        target_link_libraries(${SAMPLE_BINARY}
       	    ${PROJECT_NAME}
   	    Vulkan::Vulkan
	    SDL3::SDL3
    	    #fastgltf::fastgltf
	    glm::glm
	    VMA
	    STB
    	    #IMGUI
	    m
        )
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            target_link_libraries(${SAMPLE_BINARY} -fsanitize=undefined)
        endif()
    endforeach()
endif()
