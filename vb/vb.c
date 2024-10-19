#include <stdlib.h>
#include <vbc.h>
#include <math.h>
#include <vulkan/vulkan_core.h>

//
// HELPERS START
//

static inline char* readf(const char* filepath, int nul_terminate, size_t* size) {
    FILE* file = fopen(filepath, "rb");
    if(!file) return NULL;
    fseek(file, 0L, SEEK_END);
    long f_size = ftell(file);
    *size = f_size;
    rewind(file);
    long m_size = f_size;
    if(nul_terminate) ++m_size;
    char* content = (char*)malloc(m_size);
    if(!content) {
        fclose(file);
	return NULL;
    }
    if(fread(content, f_size, 1, file) != 1) {
        fclose(file);
	free(content);
	return NULL;
    }
    fclose(file);
    if(nul_terminate) content[f_size] = '\0';
    return content;
}

//
// HELPERS END
//

//
// VBContext START
//

static inline bool test_for_vulkan_validation_layers() {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, NULL);
    VkLayerProperties layers[count];
    vkEnumerateInstanceLayerProperties(&count, layers);
    for(size_t i = 0; i < count; i++) if(strcmp(layers[i].layerName, VB_VULKAN_VALIDATION_LAYER_NAME) == 0) return true;
    return false;
}

static inline bool test_for_vulkan_portability_extension() {
    uint32_t count;
    vkEnumerateInstanceExtensionProperties(NULL, &count, NULL);
    VkExtensionProperties extensions[count];
    vkEnumerateInstanceExtensionProperties(NULL, &count, extensions);
    for(size_t i = 0; i < count; i++) if(strcmp(extensions[i].extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) return true;
    return false;
}

static inline VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
    fprintf(stderr, "%s\n", data->pMessage);
    return VK_FALSE;
}

static inline VkDebugUtilsMessengerCreateInfoEXT fill_vulkan_debug_messenger_create_info() {
    VkDebugUtilsMessengerCreateInfoEXT info = {
	.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
    	.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
 	.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT,
	.pfnUserCallback = vulkan_debug_callback,
    };
    return info;
}

static inline void create_vulkan_debug_messenger(VBContext* ctx) {
    VkDebugUtilsMessengerCreateInfoEXT info = fill_vulkan_debug_messenger_create_info();
    PFN_vkCreateDebugUtilsMessengerEXT create_debug_utils_messenger = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(ctx->instance, "vkCreateDebugUtilsMessengerEXT");
    VB_ASSERT(create_debug_utils_messenger);
    VB_ASSERT(create_debug_utils_messenger(ctx->instance, &info, NULL, &ctx->debug_messenger) == VK_SUCCESS);
}

static inline void create_vulkan_instance(VBContext* ctx) {
    uint32_t extension_count = 0;
    const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
    VB_ASSERT(sdl_extensions);
    uint32_t new_count = extension_count;
    bool portability_extension = test_for_vulkan_portability_extension();
#ifndef NDEBUG
    if(ctx->validation_layers_support) new_count++;
#endif
    if(portability_extension) new_count++;
    const char* extensions[new_count];
    size_t i = 0;
    for(i = 0; i < extension_count; i++) extensions[i] = sdl_extensions[i];
    if(portability_extension) extensions[i] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME, i++;
#ifndef NDEBUG
    if(ctx->validation_layers_support) extensions[i] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkDebugUtilsMessengerCreateInfoEXT debug_info = fill_vulkan_debug_messenger_create_info();
#endif
    VkApplicationInfo app_info = {
	.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	.pApplicationName = ctx->info.title,
	.apiVersion = VB_VULKAN_API_VERSION,
    };
    VkInstanceCreateInfo info = {
	.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
	.flags = portability_extension ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0,
	.pApplicationInfo = &app_info,
	.enabledExtensionCount = new_count,
	.ppEnabledExtensionNames = extensions,
    };
#ifndef NDEBUG
    const char* layers[1] = {VB_VULKAN_VALIDATION_LAYER_NAME};
    if(ctx->validation_layers_support) {
	info.enabledLayerCount = 1;
	info.ppEnabledLayerNames = layers;
	info.pNext = &debug_info;
    }
#endif
    VB_ASSERT(vkCreateInstance(&info, NULL, &ctx->instance) == VK_SUCCESS);
}

static inline void pick_vulkan_physical_device(VBContext* ctx) {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, NULL);
    VB_ASSERT(device_count);
    VkPhysicalDevice devices[device_count];
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices);
    ctx->physical_device = VK_NULL_HANDLE;
    for(size_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(devices[i], &properties);
        if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ctx->physical_device = devices[i];
    }
    if(ctx->physical_device == VK_NULL_HANDLE) ctx->physical_device = devices[0];
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &properties);
    fprintf(stderr, "Picked %s as VkPhysicalDevice\n", properties.deviceName);
}

static inline void create_vulkan_device(VBContext* ctx) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_family_count, NULL);
    VkQueueFamilyProperties queue_families[queue_family_count];
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_family_count, queue_families);
    int64_t queue_idx[3] = {-1};
    for(size_t i = 0; i < queue_family_count; i++) {
	if(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) queue_idx[0] = i;
	if(queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) queue_idx[1] = i;
	VkBool32 present = false;
    	vkGetPhysicalDeviceSurfaceSupportKHR(ctx->physical_device, i, ctx->surface, &present);
	if(present) queue_idx[2] = i;
    }
    for(size_t i = 0; i < 3; i++) VB_ASSERT(queue_idx[i] != -1)
    ctx->queues_info.graphics_index = queue_idx[0];
    ctx->queues_info.compute_index = queue_idx[1];
    ctx->queues_info.present_index = queue_idx[2];
    uint32_t queue_count = 0;
    if(queue_idx[0] != queue_idx[1]) queue_count++;
    if(queue_idx[0] != queue_idx[2]) queue_count++;
    if(queue_idx[1] != queue_idx[2]) queue_count++;
    ctx->swapchain_support_data.queue_family_indices_count = queue_count;
    printf("%ld %ld %ld %d", queue_idx[0], queue_idx[1], queue_idx[2], queue_count);
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_infos[ctx->swapchain_support_data.queue_family_indices_count];
    for(size_t i = 0; i < ctx->swapchain_support_data.queue_family_indices_count; i++) {
	VkDeviceQueueCreateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	    .queueFamilyIndex = queue_idx[i],
	    .queueCount = 1,
	    .pQueuePriorities = &priority,
	};
	queue_infos[i] = info;
    }
    VkPhysicalDeviceVulkan13Features vk13features = ctx->info.vk13features;
    vk13features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceVulkan12Features vk12features = ctx->info.vk12features;
    vk12features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12features.pNext = &vk13features;
    VkPhysicalDeviceVulkan11Features vk11features = ctx->info.vk11features;
    vk11features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vk11features.pNext = &vk12features;
    VkPhysicalDeviceFeatures vk10features = ctx->info.vk10features;
    VkPhysicalDeviceFeatures2 features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &vk11features,
        .features = vk10features,
    };
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(ctx->physical_device, NULL, &count, NULL);
    VkExtensionProperties extensions[count];
    vkEnumerateDeviceExtensionProperties(ctx->physical_device, NULL, &count, extensions);
    size_t required_available = 0;
    if(ctx->info.required_extensions_count != 0) {
	for(size_t i = 0; i < ctx->info.required_extensions_count; i++) {
	    for(size_t j = 0; j < count; j++) {
		if(strcmp(extensions[j].extensionName, ctx->info.required_extensions[i]) == 0) {
		    fprintf(stderr, "Required extension %s available\n", ctx->info.required_extensions[i]);
		    required_available++;
		}
	    }
	}
    }
    VB_ASSERT(required_available == ctx->info.required_extensions_count);
    size_t optional_available = 0;
    if(ctx->info.optional_extensions_count != 0) {
	for(size_t i = 0; i < ctx->info.optional_extensions_count; i++) {
	    for(size_t j = 0; j < count; j++) {
		if(strcmp(extensions[j].extensionName, ctx->info.optional_extensions[i]) == 0) {
		    fprintf(stderr, "Optional extension %s available\n", ctx->info.optional_extensions[i]);
		    optional_available++;
		}
	    }
	}
    }
    size_t requested_available = optional_available + required_available + 1;
    const char* requested_extensions[requested_available];
    if(ctx->info.required_extensions_count != 0) {
	for(size_t i = 0; i < required_available; i++) requested_extensions[i] = ctx->info.required_extensions[i];
    }
    if(ctx->info.optional_extensions_count != 0) {
	for(size_t i = 0, x = required_available; i < ctx->info.optional_extensions_count; i++) {
	    for(size_t j = 0; j < count; j++) {
		if(strcmp(extensions[j].extensionName, ctx->info.optional_extensions[i]) == 0) {
		    requested_extensions[x] = ctx->info.optional_extensions[i];
		    x++;
		}
	    }
	}
    }
    requested_extensions[requested_available-1] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &vk12features,
        .queueCreateInfoCount = ctx->swapchain_support_data.queue_family_indices_count,
        .pQueueCreateInfos = queue_infos,
        .enabledExtensionCount = requested_available,
        .ppEnabledExtensionNames = requested_extensions,
        .pEnabledFeatures = &features.features,
    };
#ifndef NDEBUG
    const char* layer[1] = {VB_VULKAN_VALIDATION_LAYER_NAME};
    if(ctx->validation_layers_support) {
        info.ppEnabledLayerNames = layer;
        info.enabledLayerCount = 1;
    }
#endif
    VB_ASSERT(vkCreateDevice(ctx->physical_device, &info, NULL, &ctx->device) == VK_SUCCESS);
    vkGetDeviceQueue(ctx->device, ctx->queues_info.graphics_index, 0, &ctx->queues_info.graphics_queue);
    vkGetDeviceQueue(ctx->device, ctx->queues_info.compute_index, 0, &ctx->queues_info.compute_queue);
}

static inline void create_vulkan_swapchain(VBContext* ctx) {
    VkSurfaceCapabilitiesKHR surface_capabilities;
    VkExtent2D extent;
    VkPresentModeKHR present_mode;
    VkSurfaceFormatKHR format;
    uint32_t image_count;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx->physical_device, ctx->surface, &surface_capabilities);
    uint32_t format_count = 0, present_mode_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->physical_device, ctx->surface, &format_count, NULL);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx->physical_device, ctx->surface, &present_mode_count, NULL);
    VB_ASSERT(format_count && present_mode_count);
    VkSurfaceFormatKHR formats[format_count];
    VkPresentModeKHR present_modes[present_mode_count];
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->physical_device, ctx->surface, &format_count, formats);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx->physical_device, ctx->surface, &present_mode_count, present_modes);
    if(ctx->info.surface_format.format == 0) ctx->info.surface_format.format = VK_FORMAT_B8G8R8A8_SRGB;
    for(size_t i = 0; i < format_count; i++) {
        if(formats[i].format == ctx->info.surface_format.format && formats[i].colorSpace == ctx->info.surface_format.colorSpace) {
	    format = formats[i];
	    break;
        }
    }
    for(size_t i = 0; i < present_mode_count; i++) {
        if(present_modes[i] == ctx->info.present_mode) {
            present_mode = present_modes[i];
            break;
        }
        present_mode = VK_PRESENT_MODE_FIFO_KHR;
    }
    if(surface_capabilities.currentExtent.width != UINT32_MAX) {
        extent = surface_capabilities.currentExtent;
    } else {
        extent.width = VB_CLAMP(ctx->info.width,
    	    surface_capabilities.minImageExtent.width, 
    	    surface_capabilities.maxImageExtent.width);
        extent.height = VB_CLAMP(ctx->info.height,
    	    surface_capabilities.minImageExtent.height,
    	    surface_capabilities.maxImageExtent.height);
    }
    image_count = surface_capabilities.minImageCount + 1;
    if(surface_capabilities.maxImageCount > 0 && image_count > surface_capabilities.maxImageCount)
        image_count = surface_capabilities.maxImageCount;
    uint32_t indices[ctx->swapchain_support_data.queue_family_indices_count];
    if(ctx->swapchain_support_data.queue_family_indices_count == 2) {
	indices[0] = ctx->queues_info.graphics_index;
	indices[1] = ctx->queues_info.compute_index;
    } else if(ctx->swapchain_support_data.queue_family_indices_count == 1) {
	indices[0] = ctx->queues_info.graphics_index;
    }
    VkSwapchainCreateInfoKHR info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = ctx->surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = ctx->swapchain_support_data.queue_family_indices_count > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = ctx->swapchain_support_data.queue_family_indices_count,
        .pQueueFamilyIndices = indices,
        .preTransform = surface_capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
    };
    VB_ASSERT(vkCreateSwapchainKHR(ctx->device, &info, NULL, &ctx->swapchain) == VK_SUCCESS);
    vkGetSwapchainImagesKHR(ctx->device, ctx->swapchain, &image_count, NULL);
    ctx->swapchain_images = malloc(sizeof(VkImage) * image_count);
    VB_ASSERT(ctx->swapchain_images);
    vkGetSwapchainImagesKHR(ctx->device, ctx->swapchain, &image_count, ctx->swapchain_images);
    ctx->swapchain_format = format.format;
    ctx->swapchain_extent = extent;
    ctx->swapchain_support_data.format = format;
    ctx->swapchain_support_data.present_mode = present_mode;
    ctx->swapchain_support_data.surface_capabilities = surface_capabilities;
    ctx->swapchain_support_data.image_count = image_count;
    ctx->swapchain_support_data.image_sharing_mode = ctx->swapchain_support_data.queue_family_indices_count > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
}

static inline void create_vulkan_swapchain_imageviews(VBContext* ctx) {
    ctx->swapchain_image_views = malloc(sizeof(VkImageView) * ctx->swapchain_support_data.image_count);
    VB_ASSERT(ctx->swapchain_image_views);
    for(size_t i = 0; i < ctx->swapchain_support_data.image_count; i++) {
        VkImageViewCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = ctx->swapchain_images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = ctx->swapchain_format,
            .components = {
		VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
		VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.levelCount = 1,
		.layerCount = 1,
            },
        };
        VB_ASSERT(vkCreateImageView(ctx->device, &info, NULL, &ctx->swapchain_image_views[i]) == VK_SUCCESS);
    }
}

static inline void destroy_vulkan_swapchain(VBContext* ctx) {
    for(size_t i = 0; i < ctx->swapchain_support_data.image_count; i++)
	vkDestroyImageView(ctx->device, ctx->swapchain_image_views[i], NULL);
    vkDestroySwapchainKHR(ctx->device, ctx->swapchain, NULL);
}

void vb_recreate_swapchain(VBContext* ctx) {
    vkDeviceWaitIdle(ctx->device);
    int w,h;
    SDL_GetWindowSize(ctx->window, &w, &h);
    if(ctx->swapchain_support_data.surface_capabilities.currentExtent.width != UINT32_MAX) {
        ctx->swapchain_extent = ctx->swapchain_support_data.surface_capabilities.currentExtent;
    } else {
        ctx->swapchain_extent.width = VB_CLAMP(w,
    	    ctx->swapchain_support_data.surface_capabilities.minImageExtent.width, 
    	    ctx->swapchain_support_data.surface_capabilities.maxImageExtent.width);
        ctx->swapchain_extent.height = VB_CLAMP(h,
    	    ctx->swapchain_support_data.surface_capabilities.minImageExtent.height,
    	    ctx->swapchain_support_data.surface_capabilities.maxImageExtent.height);
    }
    uint32_t indices[ctx->swapchain_support_data.queue_family_indices_count];
    if(ctx->swapchain_support_data.queue_family_indices_count == 2) {
	indices[0] = ctx->queues_info.graphics_index;
	indices[1] = ctx->queues_info.compute_index;
    } else if(ctx->swapchain_support_data.queue_family_indices_count == 1) {
	indices[0] = ctx->queues_info.graphics_index;
    }
    VkSwapchainCreateInfoKHR info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = ctx->surface,
        .minImageCount = ctx->swapchain_support_data.image_count,
        .imageFormat = ctx->swapchain_support_data.format.format,
        .imageColorSpace = ctx->swapchain_support_data.format.colorSpace,
        .imageExtent = ctx->swapchain_extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = ctx->swapchain_support_data.image_sharing_mode,
        .queueFamilyIndexCount = ctx->swapchain_support_data.queue_family_indices_count,
        .pQueueFamilyIndices = indices,
        .preTransform = ctx->swapchain_support_data.surface_capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = ctx->swapchain_support_data.present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = ctx->swapchain,
    };
    VkSwapchainKHR temp_swapchain;
    VB_ASSERT(vkCreateSwapchainKHR(ctx->device, &info, NULL, &temp_swapchain) == VK_SUCCESS);
    destroy_vulkan_swapchain(ctx);
    ctx->swapchain = temp_swapchain;
    vkGetSwapchainImagesKHR(ctx->device, ctx->swapchain, &ctx->swapchain_support_data.image_count, ctx->swapchain_images);
    create_vulkan_swapchain_imageviews(ctx);
    ctx->resize = false;
}

static inline void init_frames(VBContext* ctx) {
    for(size_t i = 0; i < VB_MAX_FRAMES; i++) {
	ctx->frames[i].cmd_pool = vb_create_command_pool(ctx->device, ctx->queues_info.graphics_index, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	VB_ASSERT(ctx->frames[i].cmd_pool != VK_NULL_HANDLE);
	VkCommandBufferAllocateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	    .commandPool = ctx->frames[i].cmd_pool,
	    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	    .commandBufferCount = 1,
	};
        VB_ASSERT(vkAllocateCommandBuffers(ctx->device, &info, &ctx->frames[i].cmd_buffer) == VK_SUCCESS);
	ctx->frames[i].render_fence = vb_create_fence(ctx->device, VK_FENCE_CREATE_SIGNALED_BIT);
	VB_ASSERT(ctx->frames[i].render_fence != VK_NULL_HANDLE);
	ctx->frames[i].image_available_semaphore = vb_create_semaphore(ctx->device, 0);
	VB_ASSERT(ctx->frames[i].image_available_semaphore != VK_NULL_HANDLE);
	ctx->frames[i].finish_render_semaphore = vb_create_semaphore(ctx->device, 0);
	VB_ASSERT(ctx->frames[i].finish_render_semaphore != VK_NULL_HANDLE);
    }
}

static inline void init_vma(VBContext* ctx) {
    VmaAllocatorCreateInfo info = {
	.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
	.physicalDevice = ctx->physical_device,
	.device = ctx->device,
	.instance = ctx->instance,
    };
    VB_ASSERT(vmaCreateAllocator(&info, &ctx->allocator) == VK_SUCCESS);
}

static inline void init_cmd_dispatch(VBContext* ctx) {
    ctx->command.cmd_pool = vb_create_command_pool(ctx->device, ctx->queues_info.graphics_index, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    VB_ASSERT(ctx->command.cmd_pool != VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo info = {
	.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	.commandPool = ctx->command.cmd_pool,
	.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	.commandBufferCount = 1,
    };
    VB_ASSERT(vkAllocateCommandBuffers(ctx->device, &info, &ctx->command.cmd_buffer) == VK_SUCCESS);
    ctx->command.fence = vb_create_fence(ctx->device, 0);
    VB_ASSERT(ctx->command.fence != VK_NULL_HANDLE);
}

void vb_dispatch_command(VBContext* ctx, void (*fn)(VkCommandBuffer, void*), void* user_data) {
    vkResetFences(ctx->device, 1, &ctx->command.fence);
    vkResetCommandBuffer(ctx->command.cmd_buffer, 0);
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VB_ASSERT(vkBeginCommandBuffer(ctx->command.cmd_buffer, &begin) == VK_SUCCESS);
    fn(ctx->command.cmd_buffer, user_data);
    VB_ASSERT(vkEndCommandBuffer(ctx->command.cmd_buffer) == VK_SUCCESS);
    VkCommandBufferSubmitInfo cmd_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = ctx->command.cmd_buffer,
    };
    VkSubmitInfo2 submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmd_info,
    };
    VB_ASSERT(vkQueueSubmit2(ctx->queues_info.graphics_queue, 1, &submit, ctx->command.fence) == VK_SUCCESS);
    vkWaitForFences(ctx->device, 1, &ctx->command.fence, 1, UINT64_MAX);
}

VBContext* vb_new_context(const VBContextInfo* info) {
    VBContext* ctx = malloc(sizeof(VBContext));
    VB_ASSERT(ctx);
    ctx->info = *info;
    if(!(ctx->info.sdl3_init_flags & SDL_INIT_VIDEO)) ctx->info.sdl3_init_flags |= SDL_INIT_VIDEO;
    if(!(ctx->info.sdl3_window_flags & SDL_WINDOW_VULKAN)) ctx->info.sdl3_window_flags |= SDL_WINDOW_VULKAN;
    VB_ASSERT(SDL_Init(ctx->info.sdl3_init_flags));
    ctx->window = SDL_CreateWindow(ctx->info.title, ctx->info.width, ctx->info.height, ctx->info.sdl3_window_flags);
    VB_ASSERT(ctx->window);
    SDL_SetWindowMinimumSize(ctx->window, ctx->info.width, ctx->info.height);
    ctx->validation_layers_support = test_for_vulkan_validation_layers();
    create_vulkan_instance(ctx);
#ifndef NDEBUG
    if(ctx->validation_layers_support) create_vulkan_debug_messenger(ctx);
#endif
    pick_vulkan_physical_device(ctx);
    VB_ASSERT(SDL_Vulkan_CreateSurface(ctx->window, ctx->instance, NULL, &ctx->surface));
    create_vulkan_device(ctx);
    create_vulkan_swapchain(ctx);
    create_vulkan_swapchain_imageviews(ctx);
    init_frames(ctx);
    init_vma(ctx);
    init_cmd_dispatch(ctx);
    return ctx;
};

void vb_free_context(VBContext* ctx) {
    vkDestroyCommandPool(ctx->device, ctx->command.cmd_pool, NULL);
    vkDestroyFence(ctx->device, ctx->command.fence, NULL);
    vmaDestroyAllocator(ctx->allocator);
    for(size_t i = 0; i < VB_MAX_FRAMES; i++) {
        vkDestroyCommandPool(ctx->device, ctx->frames[i].cmd_pool, NULL);
        vkDestroyFence(ctx->device, ctx->frames[i].render_fence, NULL);
        vkDestroySemaphore(ctx->device, ctx->frames[i].image_available_semaphore, NULL);
        vkDestroySemaphore(ctx->device, ctx->frames[i].finish_render_semaphore, NULL);
    }
    destroy_vulkan_swapchain(ctx);
    vkDestroyDevice(ctx->device, NULL);
    vkDestroySurfaceKHR(ctx->instance, ctx->surface, NULL);
#ifndef NDEBUG
    if(ctx->validation_layers_support) {
        PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_utils_messenger = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(ctx->instance, "vkDestroyDebugUtilsMessengerEXT");
        destroy_debug_utils_messenger(ctx->instance, ctx->debug_messenger, NULL);
    }
#endif
    vkDestroyInstance(ctx->instance, NULL);
    SDL_DestroyWindow(ctx->window);
    SDL_Quit();
    free(ctx->swapchain_images);
    free(ctx->swapchain_image_views);
    free(ctx);
}

//
// VBContext END
//

VBBuffer* vb_new_buffer(VBContext* ctx, const size_t size, VkBufferCreateFlags usage, VmaMemoryUsage mem_usage) {
    VBBuffer* buffer = malloc(sizeof(VBBuffer));
    if(!buffer) return NULL;
    buffer->ctx = ctx;
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
    };
    VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = mem_usage,
    };
    if(vmaCreateBuffer(ctx->allocator, &buffer_info, &allocation_info, &buffer->buffer, &buffer->allocation, &buffer->info) != VK_SUCCESS) {
	free(buffer);
	return NULL;
    }
    return buffer;
}

void vb_free_buffer(VBBuffer* buffer) {
    vmaDestroyBuffer(buffer->ctx->allocator, buffer->buffer, buffer->allocation);
    free(buffer);
}

VBImage* vb_new_image(VBContext* ctx, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, bool mipmap) {
    VBImage* image = malloc(sizeof(VBImage));
    if(!image) return NULL;
    image->ctx = ctx;
    image->format = format;
    image->extent = extent;
    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = extent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
    };
    if(mipmap) image_info.mipLevels = (uint32_t)(floorf(VB_MAX(extent.width, extent.height)))+1;
    VmaAllocationCreateInfo allocation_info = {
    	.usage = VMA_MEMORY_USAGE_GPU_ONLY,
    	.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
    if(vmaCreateImage(ctx->allocator, &image_info, &allocation_info, &image->image, &image->allocation, NULL) != VK_SUCCESS) {
	free(image);
	return NULL;
    }
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    if(format == VK_FORMAT_D32_SFLOAT) aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    VkImageViewCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image->image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
	    .aspectMask = aspect,
	    .levelCount = 1,
	    .layerCount = 1,
	},
    };
    if(vkCreateImageView(ctx->device, &info, NULL, &image->image_view) != VK_SUCCESS) {
	vmaDestroyImage(ctx->allocator, image->image, image->allocation);
	free(image);
	return NULL;
    }
    return image;
}

struct copy_buffer_to_image_data {
    VBImage* image;
    VBBuffer* buffer;
};

static inline void copy_buffer_to_image(VkCommandBuffer cmd, void* user_data) {
    struct copy_buffer_to_image_data* data = (struct copy_buffer_to_image_data*)user_data;
    vb_transition_image_layout(cmd, data->image->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy copy = {
	.imageSubresource = {
	    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	    .layerCount = 1,
	},
	.imageExtent = data->image->extent,
    };
    vkCmdCopyBufferToImage(cmd, data->buffer->buffer, data->image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    vb_transition_image_layout(cmd, data->image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

}

VBImage* vb_new_image_from_data(VBContext* ctx, void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, bool mipmap) {
    size_t data_size = extent.depth * extent.width * extent.height * 4;
    VBBuffer* staging_buffer = vb_new_buffer(ctx, data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    if(!staging_buffer) return NULL;
    memcpy(staging_buffer->info.pMappedData, data, data_size);
    VBImage* image = vb_new_image(ctx, extent, format, usage, mipmap);
    if(!image) {
	vb_free_buffer(staging_buffer);
	return NULL;
    }
    struct copy_buffer_to_image_data copy_data = {image, staging_buffer};
    vb_dispatch_command(ctx, copy_buffer_to_image, &copy_data);
    vb_free_buffer(staging_buffer);
    return image;
}

void vb_free_image(VBImage* image) {
    vkDestroyImageView(image->ctx->device, image->image_view, NULL);
    vmaDestroyImage(image->ctx->allocator, image->image, image->allocation);
    free(image);
}

VkCommandPool vb_create_command_pool(VkDevice device, uint32_t queue_family_index, VkCommandPoolCreateFlags flags) {
    VkCommandPoolCreateInfo info = {
	.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	.flags = flags,
	.queueFamilyIndex = queue_family_index,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &info, NULL, &pool);
    return pool;
}

VkSemaphore vb_create_semaphore(VkDevice device, VkSemaphoreCreateFlags flags) {
    VkSemaphoreCreateInfo info = {
    	.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
	.flags = flags,
    };
    VkSemaphore semaphore = VK_NULL_HANDLE;
    vkCreateSemaphore(device, &info, NULL, &semaphore);
    return semaphore;
}

VkFence vb_create_fence(VkDevice device, VkFenceCreateFlags flags) {
    VkFenceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = flags,
    };
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device, &info, NULL, &fence);
    return fence;
}

VkShaderModule vb_create_shader_module(VkDevice device, const char* path) {
    size_t size = 0;
    char* file = readf(path, 0, &size);
    if(!file) return VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info = {
	.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	.codeSize = size,
	.pCode = (uint32_t*)file,
    };
    vkCreateShaderModule(device, &info, NULL, &module);
    if(module == VK_NULL_HANDLE) free(file);
    return module;
}

void vb_transition_image_layout(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout) {
    VkImageMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
	    .aspectMask = new_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
	    .levelCount = VK_REMAINING_MIP_LEVELS,
	    .layerCount = VK_REMAINING_ARRAY_LAYERS,
	},
    };
    VkDependencyInfo dependency = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dependency);
}
