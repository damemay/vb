#include <glm/vector_relational.hpp>
#include <set>
#include <format>
#include <fstream>
#include <SDL3/SDL_vulkan.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>
#include <vb.h>

namespace vb::fill {
    VkCommandBufferAllocateInfo cmd_buffer_allocate_info(VkCommandPool pool, uint32_t count) {
	VkCommandBufferAllocateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	    .commandPool = pool,
	    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	    .commandBufferCount = count,
	};
	return info;
    }

    VkPresentInfoKHR present_info(VkSwapchainKHR* swapchain, VkSemaphore* wait_semaphore, uint32_t* index) {
	VkPresentInfoKHR info = {
	    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
	    .waitSemaphoreCount = 1,
	    .pWaitSemaphores = wait_semaphore,
	    .swapchainCount = 1,
	    .pSwapchains = swapchain,
	    .pImageIndices = index,
	};
	return info;
    }

    VkRenderingAttachmentInfo attachment_info(VkImageView image_view, VkClearValue* clear, VkImageLayout image_layout) {
	VkRenderingAttachmentInfo info = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	    .imageView = image_view,
	    .imageLayout = image_layout,
	    .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
	    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	};
	if(clear) info.clearValue = *clear;
	return info;
    }

    VkRenderingAttachmentInfo depth_attachment_info(VkImageView image_view,
	    VkImageLayout image_layout) {
	VkRenderingAttachmentInfo info = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	    .imageView = image_view,
	    .imageLayout = image_layout,
	    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	};
	info.clearValue.depthStencil.depth = 0.0f;
	return info;
    }

    VkRenderingInfo rendering_info(VkExtent2D render_extent,
	    VkRenderingAttachmentInfo* color_attachment,
	    VkRenderingAttachmentInfo* depth_attachment) {
	VkRenderingInfo info = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
	    .renderArea = {{0, 0}, render_extent},
	    .layerCount = 1,
	    .colorAttachmentCount = 1,
	    .pColorAttachments = color_attachment,
	    .pDepthAttachment = depth_attachment,
	};
	return info;
    }

    VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspect_mask) {
	VkImageSubresourceRange range = {
	    .aspectMask = aspect_mask,
	    .levelCount = VK_REMAINING_MIP_LEVELS,
	    .layerCount = VK_REMAINING_ARRAY_LAYERS,
	};
	return range;
    }
}

namespace vb::sync {
    void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout) {
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
	    .subresourceRange = fill::image_subresource_range(new_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
	};
	VkDependencyInfo dependency = {
	    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
	    .imageMemoryBarrierCount = 1,
	    .pImageMemoryBarriers = &barrier,
	};
	vkCmdPipelineBarrier2(cmd, &dependency);
    }
}

namespace vb::create {
    std::optional<VkShaderModule> shader_module(VkDevice device, const char* path) {
	std::ifstream file {path, std::ios::ate | std::ios::binary};
	if(!file.is_open()) return std::nullopt;
	size_t size = file.tellg();
	std::vector<uint32_t> buffer (size / sizeof(uint32_t));
	file.seekg(0);
	file.read((char*)buffer.data(), size);
	file.close();
        VkShaderModuleCreateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	    .codeSize = size,
	    .pCode = buffer.data(),
        };
        VkShaderModule shader;
        if(vkCreateShaderModule(device, &info, NULL, &shader) != VK_SUCCESS) return std::nullopt;
        return shader;
    }

    VkCommandPool cmd_pool(VkDevice device, uint32_t queue_family_index,
	    VkCommandPoolCreateFlags flags) {
	VkCommandPoolCreateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    	    .flags = flags,
    	    .queueFamilyIndex = queue_family_index,
    	};
	VkCommandPool pool;
	VB_ASSERT(vkCreateCommandPool(device, &info, nullptr, &pool) == VK_SUCCESS);
	return pool;
    }

    VkSemaphore semaphore(VkDevice device, VkSemaphoreCreateFlags flags) {
        VkSemaphoreCreateInfo info = {
	   .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    	   .flags = flags,
           };
        VkSemaphore semaphore;
        VB_ASSERT(vkCreateSemaphore(device, &info, NULL, &semaphore) == VK_SUCCESS);
        return semaphore;
    }
    
    VkFence fence(VkDevice device, VkFenceCreateFlags flags) {
        VkFenceCreateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	    .flags = flags,
        };
        VkFence fence;
        VB_ASSERT(vkCreateFence(device, &info, NULL, &fence) == VK_SUCCESS);
        return fence;
    }

    VkDescriptorSetLayout descriptor_set_layout(VkDevice device,
	    std::vector<VkDescriptorSetLayoutBinding> bindings,
	    VkShaderStageFlags stages,
	    VkDescriptorSetLayoutCreateFlags flags,
	    void* next) {
	if(stages) for(auto& binding: bindings)
	    binding.stageFlags |= stages;
	VkDescriptorSetLayoutCreateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    .pNext = next,
	    .flags = flags,
	    .bindingCount = (uint32_t)bindings.size(),
	    .pBindings = bindings.data(),
	};
	VkDescriptorSetLayout set;
	VB_ASSERT(vkCreateDescriptorSetLayout(device, &info, NULL, &set) == VK_SUCCESS);
	return set;
    }
    
    VkPipeline compute_pipeline(VkDevice device,
	    VkPipelineLayout layout,
	    VkShaderModule shader_module) {
        VkPipelineShaderStageCreateInfo shader_info = {
	   .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    	   .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    	   .module = shader_module,
    	   .pName = "main",
        };
        VkComputePipelineCreateInfo info = {
	   .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    	   .stage = shader_info,
    	   .layout = layout,
        };
        VkPipeline pipeline;
        VB_ASSERT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline) == VK_SUCCESS);
        return pipeline;
    }
}

namespace vb::builder {
    void Descriptor::create(std::span<Ratio> pool_ratios, uint32_t init_sets, VkDescriptorPoolCreateFlags flags) {
	ratios.clear();
	for(auto ratio: pool_ratios) ratios.push_back(ratio);
	auto pool = create_pool(pool_ratios, init_sets, flags);
	if(!pool.has_value()) return;
	sets = init_sets * 1.5f;
	ready_pools.push_back(pool.value());
    }

    std::optional<VkDescriptorSet> Descriptor::allocate(VkDescriptorSetLayout layout, void* next) {
	auto pool = get_pool();
	if(!pool.has_value()) return std::nullopt;
	VkDescriptorSetAllocateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	    .pNext = next,
	    .descriptorPool = pool.value(),
	    .descriptorSetCount = 1,
	    .pSetLayouts = &layout,
	};
	VkDescriptorSet set;
	VkResult result = vkAllocateDescriptorSets(ctx->device, &info, &set);
	if(result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
	    full_pools.push_back(pool.value());
	    pool = get_pool();
	    if(!pool.has_value()) return std::nullopt;
	    info.descriptorPool = pool.value();
	    if(vkAllocateDescriptorSets(ctx->device, &info, &set) != VK_SUCCESS) return std::nullopt;
	}
	ready_pools.push_back(pool.value());
	return set;
    }

    void Descriptor::flush() {
	for(auto pool: ready_pools) vkResetDescriptorPool(ctx->device, pool, 0);
	for(auto pool: full_pools) {
	    vkResetDescriptorPool(ctx->device, pool, 0);
	    ready_pools.push_back(pool);
	}
	full_pools.clear();
    }

    void Descriptor::clean() {
	for(auto pool: ready_pools) vkDestroyDescriptorPool(ctx->device, pool, nullptr);
	for(auto pool: full_pools) vkDestroyDescriptorPool(ctx->device, pool, nullptr);
	ready_pools.clear();
	full_pools.clear();
    }

    std::optional<VkDescriptorPool> Descriptor::get_pool() {
	if(ready_pools.size() != 0) {
	    auto pool = ready_pools.back();
	    ready_pools.pop_back();
	    return pool;
	}
	auto pool = create_pool(ratios, sets);
	if(!pool.has_value()) return std::nullopt;
	sets *= 1.5f;
	if(sets > 4092) sets = 4092;
	return pool.value();
    }

    std::optional<VkDescriptorPool> Descriptor::create_pool(std::span<Ratio> pool_ratios, uint32_t init_sets, VkDescriptorPoolCreateFlags flags) {
	std::vector<VkDescriptorPoolSize> sizes(pool_ratios.size());
	for(size_t i = 0; i < pool_ratios.size(); i++) {
	    sizes[i].type = pool_ratios[i].type;
	    sizes[i].descriptorCount = (uint32_t)(pool_ratios[i].ratio*init_sets);
	}
	VkDescriptorPoolCreateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	    .flags = flags,
	    .maxSets = init_sets,
	    .poolSizeCount = (uint32_t)sizes.size(),
	    .pPoolSizes = sizes.data(),
	};
	VkDescriptorPool pool;
	if(vkCreateDescriptorPool(ctx->device, &info, nullptr, &pool) != VK_SUCCESS) return std::nullopt;
	return pool;
    }

    void Buffer::create(const size_t size, VkBufferCreateFlags usage, VmaMemoryUsage mem_usage) {
	VkBufferCreateInfo buffer_info = {
	    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	    .size = size,
	    .usage = usage,
	};
	VmaAllocationCreateInfo allocation_info = {
	    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
	    .usage = mem_usage,
	};

	VkBuffer temp_buffer;
	VmaAllocation temp_allocation;
	VmaAllocationInfo temp_allocation_info;
	if(vmaCreateBuffer(ctx->vma_allocator, &buffer_info, &allocation_info, &temp_buffer, &temp_allocation, &temp_allocation_info) != VK_SUCCESS) return;
	buffer = temp_buffer;
	allocation = temp_allocation;
	info = temp_allocation_info;
    }

    void Buffer::clean() {
	vmaDestroyBuffer(ctx->vma_allocator, buffer.value(), allocation.value());
    }

    void Image::create(VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, bool mipmap) {
    	this->format = format;
    	this->extent = extent;
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
	if(mipmap) image_info.mipLevels = (uint32_t)(floorf(std::max(extent.width, extent.height)))+1;
    	VmaAllocationCreateInfo allocation_info = {
    		.usage = VMA_MEMORY_USAGE_GPU_ONLY,
    		.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    	};

	VkImage temp_image;
	VmaAllocation temp_allocation;
    	if(vmaCreateImage(ctx->vma_allocator, &image_info, &allocation_info, &temp_image, &temp_allocation, nullptr) != VK_SUCCESS) return;
	image = temp_image;
	allocation = temp_allocation;

	VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
	if(format == VK_FORMAT_D32_SFLOAT) aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    	VkImageViewCreateInfo info = {
    	    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    	    .image = image.value(),
    	    .viewType = VK_IMAGE_VIEW_TYPE_2D,
    	    .format = format,
    	    .subresourceRange = {
    	       .aspectMask = aspect,
		.levelCount = 1,
		.layerCount = 1,
    	    },
    	};
	VkImageView temp_image_view;
	if(vkCreateImageView(ctx->device, &info, nullptr, &temp_image_view) != VK_SUCCESS) return;
	image_view = temp_image_view;
    }

    void Image::create(void* data, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, bool mipmap) {
	size_t data_size = extent.depth * extent.width * extent.height * 4;
	auto staging_buffer = Buffer(ctx);
	staging_buffer.create(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	memcpy(staging_buffer.info->pMappedData, data, data_size);
	create(extent, format, usage, mipmap);
	if(!all_valid()) return;
	ctx->submit_quick_command([&](VkCommandBuffer cmd) {
	    sync::transition_image(cmd, image.value(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	    VkBufferImageCopy copy = {
	        .imageSubresource = {
	            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	            .layerCount = 1,
	        },
	        .imageExtent = extent,
	    };
	    log(std::format("{}", extent.width*extent.height*extent.depth*4));
	    vkCmdCopyBufferToImage(cmd, staging_buffer.buffer.value(), image.value(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
	    sync::transition_image(cmd, image.value(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	});
	staging_buffer.clean();
    }

    void Image::clean() {
	vkDestroyImageView(ctx->device, image_view.value(), nullptr);
	vmaDestroyImage(ctx->vma_allocator, image.value(), allocation.value());
    }

    void GraphicsPipeline::add_shader(VkShaderModule& shader_module, VkShaderStageFlagBits stage) {
	VkPipelineShaderStageCreateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	    .stage = stage,
	    .module = shader_module,
	    .pName = "main",
	};
	shader_stages.push_back(info);
	shader_modules.push_back(shader_module);
    }

    void GraphicsPipeline::add_shader(const char* path, VkShaderStageFlagBits stage) {
	auto module = create::shader_module(ctx->device, path);
	add_shader(module.value(), stage);
    }

    void GraphicsPipeline::add_push_constant(const uint32_t size, VkShaderStageFlagBits stage, const uint32_t offset) {
	VkPushConstantRange range = {stage, offset, size};
	push_constants.push_back(range);
    }

    void GraphicsPipeline::create(VkRenderPass render_pass, uint32_t subpass_index, std::vector<VkDescriptorSetLayout> descriptor_layouts) {
	VkPipelineVertexInputStateCreateInfo vertex_input = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        };
        VkPipelineViewportStateCreateInfo viewport = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
	    .viewportCount = 1,
	    .scissorCount = 1,
        };
        VkPipelineColorBlendAttachmentState color_blend_attachment = {
	    .blendEnable = VK_FALSE,
	    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
		| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo color_blend = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
	    .logicOpEnable = VK_FALSE,
	    .logicOp = VK_LOGIC_OP_COPY,
	    .attachmentCount = 1,
	    .pAttachments = &color_blend_attachment,
        };
        VkDynamicState states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
	    .dynamicStateCount = 2,
	    .pDynamicStates = states,
        };
	VkPipelineLayoutCreateInfo pipeline_layout = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	    .setLayoutCount = (uint32_t)descriptor_layouts.size(),
	    .pSetLayouts = descriptor_layouts.empty() ? nullptr : descriptor_layouts.data(),
	    .pushConstantRangeCount = (uint32_t)push_constants.size(),
	    .pPushConstantRanges = push_constants.size() == 0 ? nullptr : push_constants.data(),
	};
	VkPipelineLayout temp_layout;
	if(vkCreatePipelineLayout(ctx->device, &pipeline_layout, nullptr, &temp_layout) != VK_SUCCESS) return;
	else layout = temp_layout;
        VkGraphicsPipelineCreateInfo info = {
	   .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    	   .stageCount = (uint32_t)shader_stages.size(),
    	   .pStages = shader_stages.data(),
    	   .pVertexInputState = &vertex_input,
    	   .pInputAssemblyState = &input_assembly,
    	   //.pTessellationState = tessellation,
    	   .pViewportState = &viewport,
    	   .pRasterizationState = &rasterization,
    	   .pMultisampleState = &multisample,
    	   .pDepthStencilState = &depth_stencil,
    	   .pColorBlendState = &color_blend,
    	   .pDynamicState = &dynamic_state,
    	   .layout = layout.value(),
	   .renderPass = render_pass,
	   .subpass = subpass_index,
        };
        VkPipeline temp_pipeline;
        if(vkCreateGraphicsPipelines(ctx->device, VK_NULL_HANDLE, 1, &info, NULL, &temp_pipeline) != VK_SUCCESS) return;
	pipeline = temp_pipeline;
	
	for(auto& shader: shader_modules) {
	    vkDestroyShaderModule(ctx->device, shader, nullptr);
	    shader_modules.clear();
	}
    }

    void GraphicsPipeline::clean() {
	vkDestroyPipeline(ctx->device, pipeline.value(), nullptr);
	vkDestroyPipelineLayout(ctx->device, layout.value(), nullptr);
    }
}

namespace vb {
    Context::Context(const Info& context_info): info{context_info} {
	if(!(info.sdl3_init_flags & SDL_INIT_VIDEO)) info.sdl3_init_flags |= SDL_INIT_VIDEO;
	if(!(info.sdl3_window_flags & SDL_WINDOW_VULKAN)) info.sdl3_window_flags |= SDL_WINDOW_VULKAN;
	VB_ASSERT(SDL_Init(info.sdl3_init_flags));
	window = SDL_CreateWindow(info.title.c_str(), info.width, info.height, info.sdl3_window_flags);
	VB_ASSERT(window);
	SDL_SetWindowMinimumSize(window, info.width, info.height);
	render_aspect_ratio = (float)info.height/(float)info.width;
#ifndef NDEBUG
	if(test_for_validation_layers()) validation_layers_support = true;
	else log("Running debug build without support for Vulkan validation layers");
#endif
	create_instance();
#ifndef NDEBUG
	create_debug_messenger();
#endif
	pick_physical_device();
	create_surface();
	create_device();
	create_swapchain(info.width, info.height);
	create_swapchain_image_views();
	create_frames();
	init_vma();
	init_quick_cmd();
    }

    Context::~Context() {
	vkDestroyCommandPool(device, quick_command_info.cmd_pool, nullptr);
    	vkDestroyFence(device, quick_command_info.fence, nullptr);
	vmaDestroyAllocator(vma_allocator);
	for(auto& frame: frames) {
	    vkDestroyCommandPool(device, frame.cmd_pool, nullptr);
	    vkDestroyFence(device, frame.render_fence, nullptr);
	    vkDestroySemaphore(device, frame.image_available_semaphore, nullptr);
	    vkDestroySemaphore(device, frame.finish_render_semaphore, nullptr);
	}
	destroy_swapchain();
	vkDestroyDevice(device, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
#ifndef NDEBUG
	if(validation_layers_support) {
	    auto destroy_debug_utils_messenger = (PFN_vkDestroyDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	    destroy_debug_utils_messenger(instance, debug_messenger, nullptr);
	}
#endif
	vkDestroyInstance(instance, nullptr);
	SDL_DestroyWindow(window);
	SDL_Quit();
    }

    void Context::submit_quick_command(std::function<void(VkCommandBuffer cmd)>&& fn) {
	vkResetFences(device, 1, &quick_command_info.fence);
	vkResetCommandBuffer(quick_command_info.cmd_buffer, 0);
	VkCommandBufferBeginInfo begin = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	VB_ASSERT(vkBeginCommandBuffer(quick_command_info.cmd_buffer, &begin) == VK_SUCCESS);
	fn(quick_command_info.cmd_buffer);
	VB_ASSERT(vkEndCommandBuffer(quick_command_info.cmd_buffer) == VK_SUCCESS);
	VkCommandBufferSubmitInfo cmd_info = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
	    .commandBuffer = quick_command_info.cmd_buffer,
	};
 	VkSubmitInfo2 submit = {
 	    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
	    .commandBufferInfoCount = 1,
	    .pCommandBufferInfos = &cmd_info,
 	};
 	VB_ASSERT(vkQueueSubmit2(queues_info.graphics_queue, 1, &submit, quick_command_info.fence) == VK_SUCCESS);
	vkWaitForFences(device, 1, &quick_command_info.fence, 1, UINT64_MAX);
    }

    void Context::create_instance() {
	uint32_t extension_count = 0;
	auto sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
	VB_ASSERT(sdl_extensions);
	std::vector<const char*> extensions {sdl_extensions, sdl_extensions + extension_count};
	bool portability_ext = false;
	{
	    uint32_t count;
	    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
	    VkExtensionProperties instance_ext[count];
	    vkEnumerateInstanceExtensionProperties(nullptr, &count, instance_ext);
	    for(auto extension: instance_ext) {
		if(strcmp(extension.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
		    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		    portability_ext = true;
		}
	    }
	}
	if(validation_layers_support) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	VkApplicationInfo app = {
    	    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	    .pApplicationName = info.title.c_str(),
    	    .apiVersion = api_version,
	};
	VkInstanceCreateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
	    .flags = portability_ext ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : (VkInstanceCreateFlagBits)0,
	    .pApplicationInfo = &app,
	    .enabledExtensionCount = (uint32_t)extensions.size(),
	    .ppEnabledExtensionNames = (const char**)extensions.data(),
	};
#ifndef NDEBUG
	auto debug_info = fill_debug_messenger_create_info();
	if(validation_layers_support) {
	    info.enabledLayerCount = 1;
	    info.ppEnabledLayerNames = validation_layer_name;
	    info.pNext = &debug_info;
	}
#endif
	VB_ASSERT(vkCreateInstance(&info, nullptr, &instance) == VK_SUCCESS);
    }

    void Context::pick_physical_device() {
	uint32_t device_count = 0;
    	vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
	VB_ASSERT(device_count);
    	VkPhysicalDevice devices[device_count];
    	vkEnumeratePhysicalDevices(instance, &device_count, devices);
    	for(size_t i = 0; i < device_count; i++) {
    	    VkPhysicalDeviceProperties properties;
    	    vkGetPhysicalDeviceProperties(devices[i], &properties);
    	    if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    	        physical_device = devices[i];
    	}
    	if(physical_device == VK_NULL_HANDLE) physical_device = devices[0];
       	VkPhysicalDeviceProperties properties;
    	vkGetPhysicalDeviceProperties(physical_device, &properties);
    	log(std::format("Picked {} as GPU", properties.deviceName));
	{
	    uint32_t count = 0;
	    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr);
	    VkExtensionProperties extensions[count];
	    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, extensions);
	    available_extensions.resize(count);
	    for(size_t i = 0; i < count; i++) available_extensions[i] = extensions[i].extensionName;
	    size_t found_required = 0;
	    for(auto& required: info.required_extensions) {
		for(auto& extension: extensions) {
		    if(strcmp(extension.extensionName, required) == 0) {
			log(std::format("Required extension {} available", required));
			found_required++;
		    }
		}
	    }
	    VB_ASSERT(found_required == info.required_extensions.size());
	    requested_extensions.insert(requested_extensions.end(),
		    info.required_extensions.begin(), info.required_extensions.end());
	    for(auto& optional: info.optional_extensions) {
		for(auto& extension: extensions) {
		    if(strcmp(extension.extensionName, optional) == 0) {
			requested_extensions.push_back(optional);
			log(std::format("Optional extension {} available", optional));
		    }
		}
	    }
	}
    }

    void Context::create_surface() {
	VB_ASSERT(SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface));
    }

    void Context::create_device() {
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
	VkQueueFamilyProperties queue_families[queue_family_count];
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families);
	int64_t queue_idx[3] = {-1};
    	for(size_t i = 0; i < queue_family_count; i++) {
    	    if(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) queue_idx[0] = i;
    	    if(queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) queue_idx[1] = i;
    	    VkBool32 present = false;
    	    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &present);
    	    if(present) queue_idx[2] = i;
    	}
    	for(size_t i = 0; i < 3; i++) VB_ASSERT(queue_idx[i] != -1)
    	queues_info.graphics_index = queue_idx[0];
    	queues_info.compute_index = queue_idx[1];
    	queues_info.present_index = queue_idx[2];
	std::set<uint32_t> unique_indices = {queue_idx, queue_idx + 3};
    	float priority = 1.0f;
	std::vector<VkDeviceQueueCreateInfo> queue_infos;
	for(auto& index: unique_indices) {
    	    VkDeviceQueueCreateInfo info = {
    	        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = index,
    	        .queueCount = 1,
    	        .pQueuePriorities = &priority,
    	    };
	    queue_infos.push_back(info);
    	}
    	VkPhysicalDeviceVulkan13Features vk13features = info.vk13features;
	vk13features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	vk13features.synchronization2 = VK_TRUE;
    	VkPhysicalDeviceVulkan12Features vk12features = info.vk12features;
    	vk12features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vk12features.separateDepthStencilLayouts = VK_TRUE;
	vk12features.bufferDeviceAddress = VK_TRUE;
    	vk12features.pNext = &vk13features;
    	VkPhysicalDeviceVulkan11Features vk11features = info.vk11features;
    	vk11features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    	vk11features.pNext = &vk12features;
	VkPhysicalDeviceFeatures vk10features = info.vk10features;
    	VkPhysicalDeviceFeatures2 features = {
    	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    	    .pNext = &vk11features,
	    .features = vk10features,
    	};
	if(info.enable_all_available_features) vkGetPhysicalDeviceFeatures2(physical_device, &features);
	if(info.enable_all_available_extensions) requested_extensions = available_extensions;
    	requested_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    	VkDeviceCreateInfo info = {
    	    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    	    .pNext = &vk12features,
    	    .queueCreateInfoCount = (uint32_t)queue_infos.size(),
    	    .pQueueCreateInfos = queue_infos.data(),
    	    .enabledExtensionCount = (uint32_t)requested_extensions.size(),
    	    .ppEnabledExtensionNames = requested_extensions.data(),
    	    .pEnabledFeatures = &features.features,
    	};
#ifndef NDEBUG
    	if(validation_layers_support) {
    	    info.ppEnabledLayerNames = validation_layer_name;
    	    info.enabledLayerCount = 1;
    	}
#endif
    	VB_ASSERT(vkCreateDevice(physical_device, &info, nullptr, &device) == VK_SUCCESS);
    	vkGetDeviceQueue(device, queues_info.graphics_index, 0, &queues_info.graphics_queue);
    	vkGetDeviceQueue(device, queues_info.compute_index, 0, &queues_info.compute_queue);
    	vkGetDeviceQueue(device, queues_info.present_index, 0, &queues_info.present_queue);
    }

    void Context::create_swapchain(uint32_t width, uint32_t height) {
	VkSurfaceCapabilitiesKHR surface_capabilities;
	VkExtent2D extent;
	VkPresentModeKHR present_mode;
	VkSurfaceFormatKHR format;
	uint32_t image_count;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surface_capabilities);
    	uint32_t format_count = 0, present_mode_count = 0;
    	vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);
    	vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, nullptr);
    	VB_ASSERT(format_count || present_mode_count);
    	VkSurfaceFormatKHR formats[format_count];
    	VkPresentModeKHR present_modes[present_mode_count];
    	vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats);
    	vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, present_modes);
	for(auto surface: formats) {
	    if(surface.format == info.surface_format.format && surface.colorSpace == info.surface_format.colorSpace) {
		format = surface;
		break;
	    }
	}
    	for(size_t i = 0; i < present_mode_count; i++) {
    	    if(present_modes[i] == info.present_mode) {
    	        present_mode = present_modes[i];
    	        break;
    	    }
    	    present_mode = VK_PRESENT_MODE_FIFO_KHR;
    	}
    	if(surface_capabilities.currentExtent.width != UINT32_MAX) {
    	    extent = surface_capabilities.currentExtent;
    	} else {
    	    extent.width = std::clamp(width,
		    surface_capabilities.minImageExtent.width, 
		    surface_capabilities.maxImageExtent.width);
    	    extent.height = std::clamp(height,
		    surface_capabilities.minImageExtent.height,
		    surface_capabilities.maxImageExtent.height);
    	}
    	image_count = surface_capabilities.minImageCount + 1;
    	if(surface_capabilities.maxImageCount > 0 && image_count > surface_capabilities.maxImageCount)
    	    image_count = surface_capabilities.maxImageCount;
	std::set<uint32_t> unique_indices = {
	    queues_info.graphics_index,
	    queues_info.compute_index,
	    queues_info.present_index,
	};
	std::vector<uint32_t> indices {unique_indices.begin(), unique_indices.end()};
    	VkSwapchainCreateInfoKHR info = {
    	    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    	    .surface = surface,
    	    .minImageCount = image_count,
    	    .imageFormat = format.format,
    	    .imageColorSpace = format.colorSpace,
    	    .imageExtent = extent,
    	    .imageArrayLayers = 1,
    	    .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    	    .imageSharingMode = indices.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
    	    .queueFamilyIndexCount = (uint32_t)indices.size(),
    	    .pQueueFamilyIndices = indices.data(),
    	    .preTransform = surface_capabilities.currentTransform,
    	    .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    	    .presentMode = present_mode,
    	    .clipped = VK_TRUE,
    	};
    	VB_ASSERT(vkCreateSwapchainKHR(device, &info, nullptr, &swapchain) == VK_SUCCESS);
    	vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
	swapchain_images.resize(image_count);
    	vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data());
    	swapchain_format = format.format;
    	swapchain_extent = extent;
	swapchain_support_data = {format, present_mode, surface_capabilities, image_count,
	    indices.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE, indices};
	// render_extent.width = std::min(info.width, extent.width);
	// render_extent.height = std::min(info.height, (uint32_t)(render_aspect_ratio*(float)extent.width));
    }

    void Context::create_swapchain_image_views() {
	swapchain_image_views.resize(swapchain_images.size());
    	for(size_t i = 0; i < swapchain_images.size(); i++) {
    	    VkImageViewCreateInfo info = {
    	        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    	        .image = swapchain_images[i],
    	        .viewType = VK_IMAGE_VIEW_TYPE_2D,
    	        .format = swapchain_format,
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
    	    VB_ASSERT(vkCreateImageView(device, &info, nullptr, &swapchain_image_views[i]) == VK_SUCCESS);
    	}
    }

    void Context::destroy_swapchain() {
	for(auto& image_view : swapchain_image_views)
	    vkDestroyImageView(device, image_view, nullptr);
	vkDestroySwapchainKHR(device, swapchain, nullptr);
    }

    void Context::recreate_swapchain(std::function<void(uint32_t,uint32_t)>&& call_before_swapchain_create) {
	vkDeviceWaitIdle(device);
	int w,h;
	SDL_GetWindowSize(window, &w, &h);
	if(call_before_swapchain_create) call_before_swapchain_create(w,h);
    	if(swapchain_support_data.surface_capabilities.currentExtent.width != UINT32_MAX) {
    	    swapchain_extent = swapchain_support_data.surface_capabilities.currentExtent;
    	} else {
    	    swapchain_extent.width = std::clamp((uint32_t)w,
		    swapchain_support_data.surface_capabilities.minImageExtent.width, 
		    swapchain_support_data.surface_capabilities.maxImageExtent.width);
    	    swapchain_extent.height = std::clamp((uint32_t)h,
		    swapchain_support_data.surface_capabilities.minImageExtent.height,
		    swapchain_support_data.surface_capabilities.maxImageExtent.height);
    	}
    	VkSwapchainCreateInfoKHR info = {
    	    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    	    .surface = surface,
    	    .minImageCount = swapchain_support_data.image_count,
    	    .imageFormat = swapchain_support_data.format.format,
    	    .imageColorSpace = swapchain_support_data.format.colorSpace,
    	    .imageExtent = swapchain_extent,
    	    .imageArrayLayers = 1,
    	    .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    	    .imageSharingMode = swapchain_support_data.image_sharing_mode,
    	    .queueFamilyIndexCount = (uint32_t)swapchain_support_data.queue_family_indices.size(),
    	    .pQueueFamilyIndices = swapchain_support_data.queue_family_indices.data(),
    	    .preTransform = swapchain_support_data.surface_capabilities.currentTransform,
    	    .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    	    .presentMode = swapchain_support_data.present_mode,
    	    .clipped = VK_TRUE,
	    .oldSwapchain = swapchain,
    	};
	VkSwapchainKHR temp_swapchain;
    	VB_ASSERT(vkCreateSwapchainKHR(device, &info, nullptr, &temp_swapchain) == VK_SUCCESS);
	destroy_swapchain();
	swapchain = temp_swapchain;
    	vkGetSwapchainImagesKHR(device, swapchain, &swapchain_support_data.image_count, swapchain_images.data());
	create_swapchain_image_views();
	resize = false;
    }

    void Context::create_frames() {
	for(auto& frame: frames) {
	    frame.cmd_pool = create::cmd_pool(device, queues_info.graphics_index, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	    auto info = fill::cmd_buffer_allocate_info(frame.cmd_pool);
	    VB_ASSERT(vkAllocateCommandBuffers(device, &info, &frame.cmd_buffer) == VK_SUCCESS);
	    frame.render_fence = create::fence(device, VK_FENCE_CREATE_SIGNALED_BIT);
	    frame.image_available_semaphore = create::semaphore(device);
	    frame.finish_render_semaphore = create::semaphore(device);
	}
    }

    void Context::init_vma() {
	VmaAllocatorCreateInfo info = {
	    .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
	    .physicalDevice = physical_device,
	    .device = device,
	    .instance = instance,
	};
        VB_ASSERT(vmaCreateAllocator(&info, &vma_allocator) == VK_SUCCESS);
    }

    void Context::init_quick_cmd() {
	quick_command_info.cmd_pool = create::cmd_pool(device, queues_info.graphics_index, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);;
	auto info = fill::cmd_buffer_allocate_info(quick_command_info.cmd_pool);
	VB_ASSERT(vkAllocateCommandBuffers(device, &info, &quick_command_info.cmd_buffer) == VK_SUCCESS);
	quick_command_info.fence = create::fence(device);
    }

#ifndef NDEBUG
    bool Context::test_for_validation_layers() {
        uint32_t count;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        VkLayerProperties layers[count];
        vkEnumerateInstanceLayerProperties(&count, layers);
        for(size_t i = 0; i < count; i++) if(strcmp(layers[i].layerName, validation_layer_name[0]) == 0) return true;
        return false;
    }
    
    void Context::create_debug_messenger() {
        auto info = fill_debug_messenger_create_info();
        auto create_debug_utils_messenger = (PFN_vkCreateDebugUtilsMessengerEXT)
	    vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	VB_ASSERT(create_debug_utils_messenger);
        VB_ASSERT(create_debug_utils_messenger(instance, &info, nullptr, &debug_messenger) == VK_SUCCESS);
    }

    static inline VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
        log(data->pMessage);
        return VK_FALSE;
    }

    VkDebugUtilsMessengerCreateInfoEXT Context::fill_debug_messenger_create_info() {
        VkDebugUtilsMessengerCreateInfoEXT info = {
	   .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
    	   .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
    	       VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
    	       VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    	       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
    	   .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
    	       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
    	       VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT,
    	   .pfnUserCallback = debug_callback,
        };
        return info;
    }
#endif
}
