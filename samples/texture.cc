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
    VkPhysicalDeviceFeatures vk10features = {
	.samplerAnisotropy = VK_TRUE,
    };
    auto info = vb::Context::Info {
	.title = "vbc",
	.width = 800,
	.height = 600,
	.vk10features = vk10features,
    };
    auto vbc = std::make_unique<vb::Context>(info);

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

    VkAttachmentDescription color_attachment {
	.format = vbc->swapchain_format,
	.samples = VK_SAMPLE_COUNT_1_BIT,
	.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    VkAttachmentReference color_attachment_ref {
	.attachment = 0,
	.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkAttachmentDescription depth_attachment {
	.format = VK_FORMAT_D32_SFLOAT,
	.samples = VK_SAMPLE_COUNT_1_BIT,
	.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
	.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
	.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	.finalLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
    };
    VkAttachmentReference depth_attachment_ref {
	.attachment = 1,
	.layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass {
	.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
	.colorAttachmentCount = 1,
	.pColorAttachments = &color_attachment_ref,
	.pDepthStencilAttachment = &depth_attachment_ref,
    };

    VkAttachmentDescription attachments[2] = {color_attachment, depth_attachment};
    VkRenderPassCreateInfo render_pass_info {
	.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
	.attachmentCount = 2,
	.pAttachments = attachments,
	.subpassCount = 1,
	.pSubpasses = &subpass,
    };
    VkRenderPass render_pass;
    VB_ASSERT(vkCreateRenderPass(vbc->device, &render_pass_info, nullptr, &render_pass) == VK_SUCCESS);

    graphics_pipeline.create(render_pass, 0, {layout});
    VB_ASSERT(graphics_pipeline.all_valid());

    VkFramebuffer framebuffers[vbc->swapchain_image_views.size()];
    for(size_t i = 0; i < vbc->swapchain_image_views.size(); i++) {
	VkImageView attachments[2] = {vbc->swapchain_image_views[i], depth_image.image_view};
	VkFramebufferCreateInfo framebuffer_info {
	    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
    	    .renderPass = render_pass,
    	    .attachmentCount = 2,
	    .pAttachments = attachments,
    	    .width = vbc->swapchain_extent.width,
    	    .height = vbc->swapchain_extent.height,
    	    .layers = 1,
    	};
    	VB_ASSERT(vkCreateFramebuffer(vbc->device, &framebuffer_info, nullptr, &framebuffers[i]) == VK_SUCCESS);
    }

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
	// if(vbc->resize) vbc->recreate_swapchain(render_pass);

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
	VkRenderPassBeginInfo render_begin {
	    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
	    .renderPass = render_pass,
	    .framebuffer = framebuffers[image_index],
	    .renderArea = {{0,0}, vbc->swapchain_extent},
	    .clearValueCount = 2,
	    .pClearValues = color,
	};
	vkCmdBeginRenderPass(frame->cmd_buffer, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
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
	vkCmdEndRenderPass(frame->cmd_buffer);

	VB_ASSERT(vkEndCommandBuffer(frame->cmd_buffer) == VK_SUCCESS);
 	VkPipelineStageFlags wait[1] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
 	VkSubmitInfo submit = {
 	    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
 	    .waitSemaphoreCount = 1,
 	    .pWaitSemaphores = &frame->image_available_semaphore,
 	    .pWaitDstStageMask = wait,
 	    .commandBufferCount = 1,
 	    .pCommandBuffers = &frame->cmd_buffer,
 	    .signalSemaphoreCount = 1,
 	    .pSignalSemaphores = &frame->finish_render_semaphore,
 	};
 	VB_ASSERT(vkQueueSubmit(vbc->queues_info.graphics_queue, 1, &submit, frame->render_fence) == VK_SUCCESS);
 
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

    vkDestroyRenderPass(vbc->device, render_pass, nullptr);
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
    for(size_t i=0; i<vbc->swapchain_image_views.size(); i++) vkDestroyFramebuffer(vbc->device, framebuffers[i], nullptr);
}
