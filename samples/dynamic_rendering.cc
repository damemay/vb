#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <SDL3/SDL_events.h>
#include <memory>
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

struct Rectangle {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vb::builder::Buffer vertex_buffer;
    vb::builder::Buffer index_buffer;
    VkDeviceAddress vertex_buffer_address;

    Rectangle(vb::Context* context, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
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
    	    context->submit_quick_command([&](VkCommandBuffer cmd) {
    	        VkBufferCopy copy = { .size = vertices_size };
    	        vkCmdCopyBuffer(cmd, staging_buffer.buffer, vertex_buffer.buffer, 1, &copy);
    	        VkBufferCopy copy2 = { .srcOffset = vertices_size, .size = indices_size };
    	        vkCmdCopyBuffer(cmd, staging_buffer.buffer, index_buffer.buffer, 1, &copy2);
    	    });
    	    staging_buffer.clean();
    }
};

int main(int argc, char** argv) {
    VkPhysicalDeviceFeatures vk10features { .samplerAnisotropy = VK_TRUE };
    VkPhysicalDeviceVulkan12Features vk12features { .imagelessFramebuffer = VK_TRUE };
    VkPhysicalDeviceVulkan13Features vk13features { .dynamicRendering = VK_TRUE };
    auto info = vb::Context::Info {
	.title = "vbc",
	.width = 800,
	.height = 600,
	.vk10features = vk10features,
	.vk12features = vk12features,
	.vk13features = vk13features,
    };
    auto vbc = std::make_unique<vb::Context>(info);

    auto available_extensions = vbc->get_available_extensions();
    auto enabled_extensions = vbc->get_enabled_extensions();
    vb::log("available extensions:");
    for(auto& ext: available_extensions) vb::log(ext);
    vb::log("enabled extensions:");
    for(auto& ext: enabled_extensions) vb::log(ext);

    int tw, th, tc;
    stbi_uc* data = stbi_load("../textures/texture.jpg", &tw, &th, &tc, STBI_rgb_alpha);
    VB_ASSERT(data);
    auto texture = vb::builder::Image(vbc.get());
    VkExtent3D size = {(uint32_t)tw,(uint32_t)th,1};
    texture.create(data, size);
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
    Rectangle rectangle {vbc.get(), vertices, indices};
    const std::vector<Vertex> vertices2 = {
    	{{-0.5f, -0.5f, -0.5f}, 1.0f, {1.0f, 1.0f, 1.0f}, 0.0f, {1.0f, 0.0f, 0.0f, 1.0f}},
	{{0.5f,  -0.5f, -0.5f}, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f, {0.0f, 1.0f, 0.0f, 1.0f}},
    	{{0.5f,  0.5f,  -0.5f}, 0.0f, {1.0f, 1.0f, 1.0f}, 1.0f, {0.0f, 0.0f, 1.0f, 1.0f}},
    	{{-0.5f, 0.5f,  -0.5f}, 1.0f, {1.0f, 1.0f, 1.0f}, 1.0f, {1.0f, 1.0f, 0.0f, 1.0f}}
    };
    Rectangle rectangle2 {vbc.get(), vertices2, indices};

    VkDescriptorSetLayoutBinding binding {
	.binding = 0,
	.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	.descriptorCount = 1,
	.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkDescriptorSetLayoutCreateInfo layout_info {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	.bindingCount = 1,
	.pBindings = &binding,
    };
    VkDescriptorSetLayout layout;
    VB_ASSERT(vkCreateDescriptorSetLayout(vbc->device, &layout_info, nullptr, &layout) == VK_SUCCESS);
    auto descriptor_builder = vb::builder::Descriptor(vbc.get());
    vb::builder::Descriptor::Ratio sizes[1] {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}};
    descriptor_builder.create(sizes);
    auto set = descriptor_builder.allocate(layout);
    VB_ASSERT(set);

    VkDescriptorImageInfo image_info {
	.sampler = sampler,
	.imageView = texture.image_view,
	.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkWriteDescriptorSet descriptor_write {
	.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	.dstSet = set,
	.dstBinding = 0,
	.dstArrayElement = 0,
	.descriptorCount = 1,
	.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	.pImageInfo = &image_info,
    };
    vkUpdateDescriptorSets(vbc->device, 1, &descriptor_write, 0, nullptr);

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

    graphics_pipeline.create(&rendering_info, {layout});
    VB_ASSERT(graphics_pipeline.all_valid());

    bool running = true;
    SDL_Event event;
    while(running) {
	while(SDL_PollEvent(&event) != 0) {
	    switch(event.type) {
		case SDL_EVENT_QUIT:
		    running = false;
		    break;
		case SDL_EVENT_WINDOW_RESIZED: case SDL_EVENT_WINDOW_MAXIMIZED:
		case SDL_EVENT_WINDOW_ENTER_FULLSCREEN: case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
		    vbc->resize = true;
		    break;
		case SDL_EVENT_WINDOW_HIDDEN: case SDL_EVENT_WINDOW_MINIMIZED: case SDL_EVENT_WINDOW_OCCLUDED:
		    SDL_WaitEvent(&event);
		    break;
	    }
	}

	if(vbc->resize) {
	    vbc->recreate_swapchain([&](uint32_t,uint32_t) {
		depth_image.clean();
	    });
	    depth_image.create({vbc->swapchain_extent.width, vbc->swapchain_extent.height, 1},
		VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	auto frame = vbc->get_current_frame();
	uint32_t image_index = 0;
	{
	    auto image = vbc->wait_on_image_reset_fence(frame);
	    if(!image.has_value()) continue;
	    image_index = image.value();
	}

 	vkResetCommandBuffer(frame->cmd_buffer, 0);
 	VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
 	VB_ASSERT(vkBeginCommandBuffer(frame->cmd_buffer, &begin) == VK_SUCCESS);

	VkClearValue color[2] {
	    {.color = {0.0f, 0.0f, 0.0f, 1.0f}},
	    {.depthStencil = {1.0f, 0}},
	};
	vb::sync::transition_image(frame->cmd_buffer, vbc->swapchain_images[image_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	vb::sync::transition_image(frame->cmd_buffer, depth_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
	VkRenderingAttachmentInfo color_info = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	    .imageView = vbc->swapchain_image_views[image_index],
	    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	    .clearValue = color[0],
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
	vkCmdBeginRendering(frame->cmd_buffer, &render_info);
	vkCmdBindPipeline(frame->cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline.pipeline);
	VkViewport viewport {0.0f, 0.0f, (float)vbc->swapchain_extent.width, (float)vbc->swapchain_extent.height};
	vkCmdSetViewport(frame->cmd_buffer, 0, 1, &viewport);
	VkRect2D scissor {{0,0}, vbc->swapchain_extent};
	vkCmdSetScissor(frame->cmd_buffer, 0, 1, &scissor);

	vkCmdBindDescriptorSets(frame->cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline.layout, 0, 1, &set, 0, nullptr);
	// glm::mat4 model = glm::mat4(1);
	// model = glm::rotate(model, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)vbc->swapchain_extent.width/(float)vbc->swapchain_extent.height, 0.1f, 100.0f);
	proj[1][1] *= -1;
	auto push_constants = PushConstants{proj * view, rectangle.vertex_buffer_address};
	vkCmdPushConstants(frame->cmd_buffer, graphics_pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &push_constants);
	vkCmdBindIndexBuffer(frame->cmd_buffer, rectangle.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(frame->cmd_buffer, (uint32_t)indices.size(), 2, 0, 0, 0);
	push_constants = PushConstants{proj * view, rectangle2.vertex_buffer_address};
	vkCmdPushConstants(frame->cmd_buffer, graphics_pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &push_constants);
	vkCmdBindIndexBuffer(frame->cmd_buffer, rectangle2.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(frame->cmd_buffer, (uint32_t)indices.size(), 2, 0, 0, 0);
	vkCmdEndRendering(frame->cmd_buffer);
	vb::sync::transition_image(frame->cmd_buffer, vbc->swapchain_images[image_index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

	VB_ASSERT(vkEndCommandBuffer(frame->cmd_buffer) == VK_SUCCESS);
	VkCommandBufferSubmitInfo cmd_info = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
	    .commandBuffer = frame->cmd_buffer,
	};
	VkSemaphoreSubmitInfo wait_info = {
	    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
	    .semaphore = frame->image_available_semaphore,
	    .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
	};
	VkSemaphoreSubmitInfo signal_info = {
	    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
	    .semaphore = frame->finish_render_semaphore,
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
 	VB_ASSERT(vkQueueSubmit2(vbc->queues_info.graphics_queue, 1, &submit, frame->render_fence) == VK_SUCCESS);
 
 	VkPresentInfoKHR present = {
 	    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
 	    .waitSemaphoreCount = 1,
 	    .pWaitSemaphores = &frame->finish_render_semaphore,
 	    .swapchainCount = 1,
 	    .pSwapchains = &vbc->swapchain,
 	    .pImageIndices = &image_index,
 	};
 	vkQueuePresentKHR(vbc->queues_info.graphics_queue, &present);

	vbc->frame_index++;
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
    descriptor_builder.clean();
    depth_image.clean();
}
