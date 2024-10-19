#ifndef VULKANBOILERSTRAP_H
#define VULKANBOILERSTRAP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdio.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#ifndef NDEBUG
#define VB_ASSERT(FN) assert(FN);
#else
#define VB_ASSERT(FN) if(!(FN)) { fprintf(stderr, #FN " assertion failed. Exiting!\n"; exit(1); }
#endif
#define VB_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define VB_MIN(x, y) (((x) < (y)) ? (x) : (y))
#define VB_CLAMP(x, lower, upper) (VB_MIN(upper, VB_MAX(x, lower)))

#define VB_VULKAN_API_VERSION VK_API_VERSION_1_3
#define VB_VULKAN_VALIDATION_LAYER_NAME "VK_LAYER_KHRONOS_validation"
#define VB_MAX_FRAMES 2

typedef struct VBFrame {
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buffer;
    VkSemaphore image_available_semaphore;
    VkSemaphore finish_render_semaphore;
    VkFence render_fence;
} VBFrame;

typedef struct VBQueues {
    VkQueue graphics_queue;
    VkQueue compute_queue;
    VkQueue present_queue;
    uint32_t graphics_index;
    uint32_t compute_index;
    uint32_t present_index;
} VBQueues;

typedef struct VBContextInfo {
    const char* title;
    uint32_t width;
    uint32_t height;
    SDL_InitFlags sdl3_init_flags;
    SDL_WindowFlags sdl3_window_flags;
    
    size_t required_extensions_count;
    const char** required_extensions;
    size_t optional_extensions_count;
    const char** optional_extensions;
    
    VkPhysicalDeviceFeatures vk10features;
    VkPhysicalDeviceVulkan11Features vk11features;
    VkPhysicalDeviceVulkan12Features vk12features;
    VkPhysicalDeviceVulkan13Features vk13features;
    
    VkSurfaceFormatKHR surface_format;
    VkPresentModeKHR present_mode;
} VBContextInfo;

typedef struct VBSwapchainSupportData {
    VkSurfaceFormatKHR format;
    VkPresentModeKHR present_mode;
    VkSurfaceCapabilitiesKHR surface_capabilities;
    uint32_t image_count;
    VkSharingMode image_sharing_mode;
    size_t queue_family_indices_count;
} VBSwapchainSupportData;

typedef struct VBCommand {
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buffer;
    VkFence fence;
} VBCommand;

typedef struct VBContext {
    VBContextInfo info;
    SDL_Window* window;
    VmaAllocator allocator;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VBQueues queues_info;
    VkDevice device;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VBSwapchainSupportData swapchain_support_data;
    VkFormat swapchain_format;
    VkExtent2D swapchain_extent;
    VkImage* swapchain_images;
    VkImageView* swapchain_image_views;
    bool resize;
    VBFrame frames[VB_MAX_FRAMES];
    uint8_t frame_index;

    VBCommand command;

    bool validation_layers_support;
#ifndef NDEBUG
    VkDebugUtilsMessengerEXT debug_messenger;
#endif
} VBContext;

typedef struct VBBuffer {
    VBContext* ctx;
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
} VBBuffer;

typedef struct VBImage {
    VBContext* ctx;
    VkImage image;
    VkImageView image_view;
    VmaAllocation allocation;
    VkExtent3D extent;
    VkFormat format;
} VBImage;

typedef struct VBDescriptorPool {

} VBDescriptorPool;

VBContext* vb_new_context(const VBContextInfo* info);
void vb_free_context(VBContext* ctx);
void vb_recreate_swapchain(VBContext* ctx);
void vb_dispatch_command(VBContext* ctx, void (*fn)(VkCommandBuffer, void*), void* user_data);

VBBuffer* vb_new_buffer(VBContext* ctx, const size_t size, VkBufferCreateFlags usage, VmaMemoryUsage mem_usage);
void vb_free_buffer(VBBuffer* buffer);

VBImage* vb_new_image(VBContext* ctx, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, bool mipmap);
VBImage* vb_new_image_from_data(VBContext* ctx, void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, bool mipmap);
void vb_free_image(VBImage* image);

VkCommandPool vb_create_command_pool(VkDevice device, uint32_t queue_family_index, VkCommandPoolCreateFlags flags);
VkSemaphore vb_create_semaphore(VkDevice device, VkSemaphoreCreateFlags flags);
VkFence vb_create_fence(VkDevice device, VkFenceCreateFlags flags);
VkShaderModule vb_create_shader_module(VkDevice device, const char* path);

void vb_transition_image_layout(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout);

#ifdef __cplusplus
}
#endif

#endif
