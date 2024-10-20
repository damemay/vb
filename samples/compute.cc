#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <SDL3/SDL_events.h>
#include <memory>
#include <format>
#include <vb.h>
#include <vulkan/vulkan_core.h>
#include <stb/stb_image.h>

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

struct PushConstants {
    glm::mat4 render_matrix;
    VkDeviceAddress vertex_buffer;
};

struct Frame {
    vb::builder::CommandPool pool;
    VkCommandBuffer cmd;
    VkSemaphore image_available;
    VkSemaphore finish_render;
    VkFence render;
};

struct Rectangle {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vb::builder::Buffer vertex_buffer;
    vb::builder::Buffer index_buffer;
    VkDeviceAddress vertex_buffer_address;

    Rectangle(vb::Context* context, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, vb::builder::CommandPool pool)
	: vertex_buffer{context}, index_buffer{context}, vertices{vertices}, indices{indices} {
	    const size_t vertices_size = sizeof(Vertex) * vertices.size();
	    vertex_buffer.create(vertices_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
		    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VMA_MEMORY_USAGE_GPU_ONLY);
	    VB_ASSERT(vertex_buffer.all_valid());
	    VkBufferDeviceAddressInfo address = {
    	        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    	        .buffer = vertex_buffer.buffer,
    	    };
    	    vertex_buffer_address = vkGetBufferDeviceAddress(context->device, &address);

	    const size_t indices_size = sizeof(uint32_t) * indices.size();
	    index_buffer.create(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT
		    | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	    VB_ASSERT(index_buffer.all_valid());

	    auto staging_buffer = vb::builder::Buffer{context};
    	    staging_buffer.create(vertices_size + indices_size,
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	    VB_ASSERT(staging_buffer.all_valid());
    	    memcpy(staging_buffer.info.pMappedData, vertices.data(), vertices_size);
    	    memcpy((char*)(staging_buffer.info.pMappedData)+vertices_size, indices.data(),
		    indices_size);
	    auto cmd = pool.allocate();
	    if(!cmd) return;
	    pool.submit_command_buffer_to_queue(cmd, [&](VkCommandBuffer cmd) {
    	        VkBufferCopy copy = { .size = vertices_size };
    	        vkCmdCopyBuffer(cmd, staging_buffer.buffer, vertex_buffer.buffer, 1, &copy);
    	        VkBufferCopy copy2 = { .srcOffset = vertices_size, .size = indices_size };
    	        vkCmdCopyBuffer(cmd, staging_buffer.buffer, index_buffer.buffer, 1, &copy2);
    	    });
    	    staging_buffer.clean();
    }
};

inline VkDeviceSize aligned_size(VkDeviceSize value, VkDeviceSize alignment) {
    return (value+alignment-1)&~(alignment-1);
}

int main(int argc, char** argv) {
    VkPhysicalDeviceDescriptorBufferFeaturesEXT buffer_ext = {
	.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
	.descriptorBuffer = VK_TRUE,
    };
    auto info = vb::Context::Info {
	.title = "vbc",
	.width = 800,
	.height = 600,
	.required_extensions = {
	    VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
	},
	.vk10features = {
	    .samplerAnisotropy = VK_TRUE,
	},
	.vk12features = {
	    .descriptorIndexing = VK_TRUE,
	    .bufferDeviceAddress = VK_TRUE,
	},
	.vk13features = {
	    .pNext = &buffer_ext,
	    .synchronization2 = VK_TRUE,
	    .dynamicRendering = VK_TRUE,
	},
    };
    auto vbc = std::make_unique<vb::Context>(info);

    auto available_extensions = vbc->get_available_extensions();
    auto enabled_extensions = vbc->get_enabled_extensions();
    vb::log("available extensions:");
    for(auto& ext: available_extensions) vb::log(std::format("\t{}",ext));
    vb::log("enabled extensions:");
    for(auto& ext: enabled_extensions) vb::log(std::format("\t{}",ext));

    auto vkGetDescriptorSetLayoutSizeEXT = (PFN_vkGetDescriptorSetLayoutSizeEXT)vkGetDeviceProcAddr(vbc->device, "vkGetDescriptorSetLayoutSizeEXT");
    auto vkGetDescriptorSetLayoutBindingOffsetEXT = (PFN_vkGetDescriptorSetLayoutBindingOffsetEXT)vkGetDeviceProcAddr(vbc->device, "vkGetDescriptorSetLayoutBindingOffsetEXT");
    auto vkGetDescriptorEXT = (PFN_vkGetDescriptorEXT)vkGetDeviceProcAddr(vbc->device, "vkGetDescriptorEXT");
    auto vkCmdSetDescriptorBufferOffsetsEXT = (PFN_vkCmdSetDescriptorBufferOffsetsEXT)vkGetDeviceProcAddr(vbc->device, "vkCmdSetDescriptorBufferOffsetsEXT");
    auto vkCmdBindDescriptorBuffersEXT = (PFN_vkCmdBindDescriptorBuffersEXT)vkGetDeviceProcAddr(vbc->device, "vkCmdBindDescriptorBuffersEXT");

    auto cmdpool = vb::builder::CommandPool(vbc.get());
    cmdpool.create(vbc->queues_info.graphics_queue, vbc->queues_info.graphics_index);
    cmdpool.all_valid();

    int tw, th, tc;
    stbi_uc* data = stbi_load("../textures/texture.jpg", &tw, &th, &tc, STBI_rgb_alpha);
    VB_ASSERT(data);
    auto texture = vb::builder::Image(vbc.get());
    VkExtent3D size = {(uint32_t)tw,(uint32_t)th,1};
    texture.create(cmdpool, data, size);
    VB_ASSERT(texture.all_valid());

    VkPhysicalDeviceProperties pdev_prop{};
    vkGetPhysicalDeviceProperties(vbc->physical_device, &pdev_prop);
    VkSamplerCreateInfo sampler_info = {
	.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
	.magFilter = VK_FILTER_LINEAR,
	.minFilter = VK_FILTER_LINEAR,
	.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
	.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	.mipLodBias = 0.0f,
	.anisotropyEnable = VK_TRUE,
	.maxAnisotropy = pdev_prop.limits.maxSamplerAnisotropy,
	.compareEnable = VK_FALSE,
	.compareOp = VK_COMPARE_OP_ALWAYS,
	.minLod = 0.0f,
	.maxLod = 0.0f,
	.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE,
	.unnormalizedCoordinates = VK_FALSE,
    };
    VkSampler sampler;
    VB_ASSERT(vkCreateSampler(vbc->device, &sampler_info, nullptr, &sampler) == VK_SUCCESS);

    const std::vector<Vertex> vertices = {
    	{{-0.5f, -0.5f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f}, 0.0f, {1.0f, 0.0f, 0.0f, 1.0f}},
	{{0.5f,  -0.5f, 0.0f}, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f, {0.0f, 1.0f, 0.0f, 1.0f}},
    	{{0.5f,  0.5f,  0.0f}, 0.0f, {1.0f, 1.0f, 1.0f}, 1.0f, {0.0f, 0.0f, 1.0f, 1.0f}},
    	{{-0.5f, 0.5f,  0.0f}, 1.0f, {1.0f, 1.0f, 1.0f}, 1.0f, {1.0f, 1.0f, 0.0f, 1.0f}}
    };
    const std::vector<uint32_t> indices = {0,1,2,2,3,0};
    Rectangle rectangle {vbc.get(), vertices, indices, cmdpool};
    const std::vector<Vertex> vertices2 = {
    	{{-0.5f, -0.5f, -0.5f}, 1.0f, {1.0f, 1.0f, 1.0f}, 0.0f, {1.0f, 0.0f, 0.0f, 1.0f}},
	{{0.5f,  -0.5f, -0.5f}, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f, {0.0f, 1.0f, 0.0f, 1.0f}},
    	{{0.5f,  0.5f,  -0.5f}, 0.0f, {1.0f, 1.0f, 1.0f}, 1.0f, {0.0f, 0.0f, 1.0f, 1.0f}},
    	{{-0.5f, 0.5f,  -0.5f}, 1.0f, {1.0f, 1.0f, 1.0f}, 1.0f, {1.0f, 1.0f, 0.0f, 1.0f}}
    };
    Rectangle rectangle2 {vbc.get(), vertices2, indices, cmdpool};

    VkDescriptorSetLayoutBinding binding {
	.binding = 0,
	.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	.descriptorCount = 1,
	.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkDescriptorSetLayoutCreateInfo layout_info {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
	.bindingCount = 1,
	.pBindings = &binding,
    };
    VkDescriptorSetLayout layout;
    VB_ASSERT(vkCreateDescriptorSetLayout(vbc->device, &layout_info, nullptr, &layout) == VK_SUCCESS);

    VkPhysicalDeviceDescriptorBufferPropertiesEXT descriptor_buffer_prop = {
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT
    };
    VkPhysicalDeviceProperties2 properties = {
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
	&descriptor_buffer_prop,
    };
    vkGetPhysicalDeviceProperties2(vbc->physical_device, &properties);
    size_t layout_size = 0;
    vkGetDescriptorSetLayoutSizeEXT(vbc->device, layout, &layout_size);
    layout_size = aligned_size(layout_size, descriptor_buffer_prop.descriptorBufferOffsetAlignment);
    size_t binding_offset = 0;
    vkGetDescriptorSetLayoutBindingOffsetEXT(vbc->device, layout, 0, &binding_offset);
    auto descriptor_buffer = vb::builder::Buffer(vbc.get());
    descriptor_buffer.create(size.depth*size.width*size.height*4*layout_size,
	    VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    VB_ASSERT(descriptor_buffer.all_valid());
    char* buffer_data = (char*)descriptor_buffer.info.pMappedData;
    VkDescriptorImageInfo image_info {
	.sampler = sampler,
	.imageView = texture.image_view,
	.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorGetInfoEXT image_desc_info {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
	.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	.data {
	    .pCombinedImageSampler = &image_info,
	},
    };
    vkGetDescriptorEXT(vbc->device, &image_desc_info, descriptor_buffer_prop.combinedImageSamplerDescriptorSize, buffer_data+0 * layout_size + binding_offset);
    VkBufferDeviceAddressInfo addr_info {
	.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
	.buffer = descriptor_buffer.buffer,
    };
    auto address = vkGetBufferDeviceAddress(vbc->device, &addr_info);
    VkDescriptorBufferBindingInfoEXT binding_info = {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
	.address = address,
	.usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
    };
    uint32_t buffer_index = 0;

    auto graphics_pipeline = vb::builder::GraphicsPipeline{vbc.get()};
    graphics_pipeline.set_front_face(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    graphics_pipeline.enable_depth_test();

    graphics_pipeline.add_shader("../shaders/full_vert.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    graphics_pipeline.add_shader("../shaders/textured.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    graphics_pipeline.add_push_constant(sizeof(PushConstants), VK_SHADER_STAGE_VERTEX_BIT);

    auto depth_image = vb::builder::Image(vbc.get());
    depth_image.create({vbc->swapchain_extent.width, vbc->swapchain_extent.height, 1},
	    VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    VB_ASSERT(depth_image.all_valid());

    VkFormat color_format[1] = {vbc->swapchain_format};
    VkPipelineRenderingCreateInfo rendering_info {
	.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
	.colorAttachmentCount = 1,
	.pColorAttachmentFormats = color_format,
	.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
    };

    graphics_pipeline.create(&rendering_info, VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT, {layout});
    VB_ASSERT(graphics_pipeline.all_valid());

    auto comp_image = vb::builder::Image(vbc.get());
    comp_image.create({vbc->swapchain_extent.width, vbc->swapchain_extent.height, 1},
	    VK_FORMAT_R16G16B16A16_SFLOAT);
    VB_ASSERT(comp_image.all_valid());

    VkDescriptorSetLayoutBinding compute_binding {
	.binding = 0,
	.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	.descriptorCount = 1,
	.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo compute_layout_info {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
	.bindingCount = 1,
	.pBindings = &compute_binding,
    };
    VkDescriptorSetLayout compute_dlayout;
    VB_ASSERT(vkCreateDescriptorSetLayout(vbc->device, &compute_layout_info, nullptr, &compute_dlayout) == VK_SUCCESS);

    size_t compute_dlayout_size = 0;
    vkGetDescriptorSetLayoutSizeEXT(vbc->device, compute_dlayout, &compute_dlayout_size);
    compute_dlayout_size = aligned_size(compute_dlayout_size, descriptor_buffer_prop.descriptorBufferOffsetAlignment);
    size_t cbinding_offset = 0;
    vkGetDescriptorSetLayoutBindingOffsetEXT(vbc->device, compute_dlayout, 0, &cbinding_offset);
    auto cdescriptor_buffer = vb::builder::Buffer(vbc.get());
    cdescriptor_buffer.create(vbc->swapchain_extent.width*vbc->swapchain_extent.height*4*compute_dlayout_size, VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    VB_ASSERT(cdescriptor_buffer.all_valid());
    char* cbuffer_data = (char*)cdescriptor_buffer.info.pMappedData;
    VkDescriptorImageInfo cimage_info {
	.imageView = comp_image.image_view,
	.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkDescriptorGetInfoEXT cimage_desc_info {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
	.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	.data {
	    .pStorageImage = &cimage_info,
	},
    };
    vkGetDescriptorEXT(vbc->device, &cimage_desc_info, descriptor_buffer_prop.storageImageDescriptorSize, cbuffer_data);
    VkBufferDeviceAddressInfo caddr_info {
	.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
	.buffer = cdescriptor_buffer.buffer,
    };
    auto caddress = vkGetBufferDeviceAddress(vbc->device, &caddr_info);
    VkDescriptorBufferBindingInfoEXT cbinding_info = {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
	.address = caddress,
	.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
    };
    uint32_t cbuffer_index = 0;

    VkPipelineLayoutCreateInfo compute_layout_inf {
	.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	.setLayoutCount = 1,
	.pSetLayouts = &compute_dlayout, 
    };
    VkPipelineLayout compute_layout;
    vkCreatePipelineLayout(vbc->device, &compute_layout_inf, nullptr, &compute_layout);
    auto comp_shader = vb::create::shader_module(vbc->device, "../shaders/grad.comp.spv");
    VB_ASSERT(comp_shader);
    VkPipelineShaderStageCreateInfo stage = {
	.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	.stage = VK_SHADER_STAGE_COMPUTE_BIT,
	.module = comp_shader,
	.pName = "main",
    };
    VkComputePipelineCreateInfo compute_info = {
	.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
	.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
	.stage = stage,
	.layout = compute_layout,
    };
    VkPipeline compute_pipeline;
    VB_ASSERT(vkCreateComputePipelines(vbc->device, 0, 1, &compute_info, nullptr, &compute_pipeline) == VK_SUCCESS);
    vkDestroyShaderModule(vbc->device, comp_shader, nullptr);

    vbc->set_resize_callback([&]() {
	vbc->recreate_swapchain([&](uint32_t,uint32_t) {
	    depth_image.clean();
	    comp_image.clean();
	});
	depth_image.create({vbc->swapchain_extent.width, vbc->swapchain_extent.height, 1},
	    VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
	comp_image.create({vbc->swapchain_extent.width, vbc->swapchain_extent.height, 1},
	    VK_FORMAT_R16G16B16A16_SFLOAT);
    });

    cmdpool.clean();

    Frame frames[2] = {{vbc.get()},{vbc.get()}};
    for(auto& frame: frames) {
	frame.pool.create(vbc->queues_info.graphics_queue, vbc->queues_info.graphics_index);
	VB_ASSERT(frame.pool.all_valid());
	frame.cmd = frame.pool.allocate();
	VB_ASSERT(frame.cmd);
	frame.render = vb::create::fence(vbc->device, VK_FENCE_CREATE_SIGNALED_BIT);
	VB_ASSERT(frame.render);
	frame.image_available = vb::create::semaphore(vbc->device);
	VB_ASSERT(frame.image_available);
	frame.finish_render = vb::create::semaphore(vbc->device);
	VB_ASSERT(frame.finish_render);
    }

    uint8_t frame_index = 0;

    bool running = true;
    SDL_Event event;
    while(running) {
	bool resize = false;
	while(SDL_PollEvent(&event) != 0) {
	    switch(event.type) {
		case SDL_EVENT_QUIT:
		    running = false;
		    break;
		case SDL_EVENT_WINDOW_RESIZED: case SDL_EVENT_WINDOW_MAXIMIZED:
		case SDL_EVENT_WINDOW_ENTER_FULLSCREEN: case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
		    resize = true;
		    break;
		case SDL_EVENT_WINDOW_HIDDEN: case SDL_EVENT_WINDOW_MINIMIZED: case SDL_EVENT_WINDOW_OCCLUDED:
		    SDL_WaitEvent(&event);
		    break;
	    }
	}

	if(resize) {
	    vbc->resize_callback();
	    resize = false;
	}

	auto frame = &frames[frame_index%2];
	vkWaitForFences(vbc->device, 1, &frame->render, VK_TRUE, UINT64_MAX);
	auto next = vbc->acquire_next_image(frame->image_available);
	if(!next.has_value()) continue;
	uint32_t image_index = next.value();
	vkResetFences(vbc->device, 1, &frame->render);
 	vkResetCommandBuffer(frame->cmd, 0);
 	VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
 	VB_ASSERT(vkBeginCommandBuffer(frame->cmd, &begin) == VK_SUCCESS);

	VkClearValue color[2] {
	    {.color = {0.0f, 0.0f, 0.0f, 1.0f}},
	    {.depthStencil = {1.0f, 0}},
	};
	vb::sync::transition_image(frame->cmd, comp_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	vkCmdBindPipeline(frame->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
	vkCmdBindDescriptorBuffersEXT(frame->cmd, 1, &cbinding_info);
	vkCmdSetDescriptorBufferOffsetsEXT(frame->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_layout, 0, 1, &cbuffer_index, &cbinding_offset);
	vkCmdDispatch(frame->cmd, ceil(vbc->swapchain_extent.width/16.0), ceil(vbc->swapchain_extent.height/16.0), 1);
	vb::sync::transition_image(frame->cmd, comp_image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	vb::sync::transition_image(frame->cmd, vbc->swapchain_images[image_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	vb::sync::blit_image(frame->cmd, comp_image.image, vbc->swapchain_images[image_index], comp_image.extent, {vbc->swapchain_extent.width, vbc->swapchain_extent.height, 1});
	vb::sync::transition_image(frame->cmd, vbc->swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	vb::sync::transition_image(frame->cmd, depth_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
	VkRenderingAttachmentInfo color_info = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	    .imageView = vbc->swapchain_image_views[image_index],
	    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	};
	VkRenderingAttachmentInfo depth_info = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	    .imageView = depth_image.image_view,
	    .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
	    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	    .clearValue = color[1],
	};
	VkRenderingInfo render_info = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
	    .renderArea = {{0,0}, vbc->swapchain_extent},
	    .layerCount = 1,
	    .colorAttachmentCount = 1,
	    .pColorAttachments = &color_info,
	    .pDepthAttachment = &depth_info,
	};
	vkCmdBeginRendering(frame->cmd, &render_info);
	vkCmdBindPipeline(frame->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline.pipeline);
	VkViewport viewport {0.0f, 0.0f, (float)vbc->swapchain_extent.width, (float)vbc->swapchain_extent.height};
	vkCmdSetViewport(frame->cmd, 0, 1, &viewport);
	VkRect2D scissor {{0,0}, vbc->swapchain_extent};
	vkCmdSetScissor(frame->cmd, 0, 1, &scissor);

	vkCmdBindDescriptorBuffersEXT(frame->cmd, 1, &binding_info);
	vkCmdSetDescriptorBufferOffsetsEXT(frame->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline.layout, 0, 1, &buffer_index, &binding_offset);
	// glm::mat4 model = glm::mat4(1);
	// model = glm::rotate(model, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)vbc->swapchain_extent.width/(float)vbc->swapchain_extent.height, 0.1f, 100.0f);
	proj[1][1] *= -1;
	auto push_constants = PushConstants{proj * view, rectangle.vertex_buffer_address};
	vkCmdPushConstants(frame->cmd, graphics_pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &push_constants);
	vkCmdBindIndexBuffer(frame->cmd, rectangle.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(frame->cmd, (uint32_t)indices.size(), 2, 0, 0, 0);
	push_constants = PushConstants{proj * view, rectangle2.vertex_buffer_address};
	vkCmdPushConstants(frame->cmd, graphics_pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &push_constants);
	vkCmdBindIndexBuffer(frame->cmd, rectangle2.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(frame->cmd, (uint32_t)indices.size(), 2, 0, 0, 0);
	vkCmdEndRendering(frame->cmd);
	vb::sync::transition_image(frame->cmd, vbc->swapchain_images[image_index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

	VB_ASSERT(vkEndCommandBuffer(frame->cmd) == VK_SUCCESS);
	VkCommandBufferSubmitInfo cmd_info = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
	    .commandBuffer = frame->cmd,
	};
	VkSemaphoreSubmitInfo wait_info = {
	    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
	    .semaphore = frame->image_available,
	    .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	};
	VkSemaphoreSubmitInfo signal_info = {
	    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
	    .semaphore = frame->finish_render,
	    .stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
	};
 	VkSubmitInfo2 submit = {
 	    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
	    .waitSemaphoreInfoCount = 1,
	    .pWaitSemaphoreInfos = &wait_info,
	    .commandBufferInfoCount = 1,
	    .pCommandBufferInfos = &cmd_info,
	    .signalSemaphoreInfoCount = 1,
	    .pSignalSemaphoreInfos = &signal_info,
 	};
 	VB_ASSERT(vkQueueSubmit2(vbc->queues_info.graphics_queue, 1, &submit, frame->render) == VK_SUCCESS);
 
 	VkPresentInfoKHR present = {
 	    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
 	    .waitSemaphoreCount = 1,
 	    .pWaitSemaphores = &frame->finish_render,
 	    .swapchainCount = 1,
 	    .pSwapchains = &vbc->swapchain,
 	    .pImageIndices = &image_index,
 	};
 	vkQueuePresentKHR(vbc->queues_info.graphics_queue, &present);

	frame_index++;
    }
    vkDeviceWaitIdle(vbc->device);

    graphics_pipeline.clean();
    rectangle.vertex_buffer.clean();
    rectangle.index_buffer.clean();
    rectangle2.vertex_buffer.clean();
    rectangle2.index_buffer.clean();
    texture.clean();
    vkDestroySampler(vbc->device, sampler, nullptr);
    vkDestroyDescriptorSetLayout(vbc->device, layout, nullptr);
    descriptor_buffer.clean();
    comp_image.clean();
    vkDestroyPipeline(vbc->device, compute_pipeline, nullptr);
    vkDestroyPipelineLayout(vbc->device, compute_layout, nullptr);
    vkDestroyDescriptorSetLayout(vbc->device, compute_dlayout, nullptr);
    cdescriptor_buffer.clean();
    depth_image.clean();
    for(auto& frame: frames) {
	frame.pool.clean();
	vkDestroySemaphore(vbc->device, frame.finish_render, nullptr);
	vkDestroySemaphore(vbc->device, frame.image_available, nullptr);
	vkDestroyFence(vbc->device, frame.render, nullptr);
    }
}
