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
find_package(SDL3 REQUIRED)

if (WIN32)
   set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
endif()

include_directories(
    vb/
    ext/
    ext/volk/
    ext/VulkanMemoryAllocator/include/
)

add_library(${PROJECT_NAME}
    vb/vb.cc
)

if(SAMPLE)
    add_subdirectory(ext/volk)
    add_library(VMA ext/vma.cc)
    add_library(STB ext/stb.cc)
    find_package(glm REQUIRED)
    set(WORKING_SAMPLES 
	samples/compute.cc
    )
    foreach(file ${WORKING_SAMPLES})
        get_filename_component(sample ${file} NAME_WLE)
        set(SAMPLE_BINARY ${sample})
        add_executable(${SAMPLE_BINARY} ${file})
        target_link_libraries(${SAMPLE_BINARY}
       	    ${PROJECT_NAME}
	    volk
	    SDL3::SDL3
	    glm::glm
	    VMA
	    STB
	    m
        )
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            target_link_libraries(${SAMPLE_BINARY} -fsanitize=undefined)
        endif()
    endforeach()
endif()
