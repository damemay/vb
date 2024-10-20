# vulkan boilerstrap

Vulkan boilerplate/bootstrap with SDL3 window tailored for my own usage written in C++20 standard with somewhat C-ish style.

It's not a one stop solution. It's made more to learn Vulkan API and it's possibilities without limiting myself to creating one kind of a rendering engine.

Nevertheless, for sanity's sake I had to make some assumption about creating VkInstance, VkPhysicalDevice, VkDevice and VkSwapchains:
- If extension is available, VkInstance is created with `VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME` extension.
- If `NDEBUG` is not defined and `VK_EXT_DEBUG_UTILS_EXTENSION_NAME` is available, debug utils and messenger are created.
- `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` is a first choice for GPU. Falls back to first GPU available in array if there's no discrete GPU. Features and extensions are checked after choosing device.
- VkDevice always asks for `VK_KHR_SWAPCHAIN` extension.
- `vb::Context::Info.surface_format` defaults to `VK_FORMAT_B8G8R8A8_SRGB` with `VK_COLOR_SPACE_SRGB_NONLINEAR_KHR`.
- `vb::Context::Info.present_mode` defaults to `VK_PRESENT_MODE_FIFO_KHR`.


Not all samples are updated to newest `vb.h`. Compilable and working samples are listed in `CMakeLists.txt` as `${WORKING_SAMPLES}`.


## Using in project
VB is built with CMake and it's trivial to include it in your own CMake project:
```cmake
add_subdirectory(vb)
target_link_libraries(${PROJECT_NAME}
    vb::vb
    volk
    SDL3::SDL3
    VMA
    STB
    m
)
```
