#include <SDL3/SDL_events.h>
#include <memory>
#include <vb.h>
#include <vulkan/vulkan_core.h>

struct Vertex {
    glm::vec4 position;
    glm::vec4 color;
};

struct PushConstants {
    VkDeviceAddress vertex_buffer;
};

int main(int argc, char** argv) {
    auto info = vb::Context::Info {
	.title = "vbc",
	.width = 800,
	.height = 600,
    };
    auto vbc = std::make_unique<vb::Context>(info);

    const std::vector<Vertex> vertices = {
    	{{0.0f, -0.5f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
	{{0.5f, 0.5f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
    	{{-0.5f, 0.5f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}}
    };
    const size_t vertices_size = sizeof(Vertex) * vertices.size();
    auto vertex_buffer = vb::create::Buffer{vbc.get()};
    vertex_buffer.create(vertices_size,
	    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    VB_ASSERT(vertex_buffer.buffer.has_value() && vertex_buffer.allocation.has_value() && vertex_buffer.info.has_value());
    VkBufferDeviceAddressInfo address = {
	.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
	.buffer = vertex_buffer.buffer.value(),
    };
    auto buffer_address = vkGetBufferDeviceAddress(vbc->device, &address);
    auto push_constants = PushConstants{buffer_address};

    auto staging_buffer = vb::create::Buffer{vbc.get()};
    staging_buffer.create(vertices_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    VB_ASSERT(staging_buffer.buffer.has_value() && staging_buffer.allocation.has_value() && staging_buffer.info.has_value());
    memcpy(staging_buffer.info->pMappedData, vertices.data(), vertices_size);
    vbc->submit_quick_command([&](VkCommandBuffer cmd) {
	VkBufferCopy copy = { .size = vertices_size };
	vkCmdCopyBuffer(cmd, staging_buffer.buffer.value(), vertex_buffer.buffer.value(), 1, &copy);
    });
    staging_buffer.clean();

    auto graphics_pipeline = vb::create::GraphicsPipeline{vbc.get()};
    graphics_pipeline.add_shader("../shaders/triangle_buffered.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    graphics_pipeline.add_shader("../shaders/triangle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    graphics_pipeline.add_push_constant(sizeof(PushConstants), VK_SHADER_STAGE_VERTEX_BIT);

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
    VkSubpassDescription subpass {
	.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
	.colorAttachmentCount = 1,
	.pColorAttachments = &color_attachment_ref,
    };
    VkRenderPassCreateInfo render_pass_info {
	.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
	.attachmentCount = 1,
	.pAttachments = &color_attachment,
	.subpassCount = 1,
	.pSubpasses = &subpass,
    };
    VkRenderPass render_pass;
    VB_ASSERT(vkCreateRenderPass(vbc->device, &render_pass_info, nullptr, &render_pass) == VK_SUCCESS);

    graphics_pipeline.create(render_pass, 0);
    VB_ASSERT(graphics_pipeline.pipeline.has_value());

    vbc->create_swapchain_framebuffers(render_pass);

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

	if(vbc->resize) vbc->recreate_swapchain(render_pass);

	auto frame = vbc->get_current_frame();
	uint32_t image_index = 0;
	{
	    auto image = vbc->wait_on_image_reset_fence(frame);
	    if(!image.has_value()) continue;
	    image_index = image.value();
	}


	vb::render::begin_reset_command_buffer(frame->cmd_buffer);

	VkClearValue color {{{0.0f, 0.0f, 0.0f, 1.0f}}};
	VkRenderPassBeginInfo render_begin {
	    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
	    .renderPass = render_pass,
	    .framebuffer = vbc->swapchain_framebuffers[image_index],
	    .renderArea = {{0,0}, vbc->swapchain_extent},
	    .clearValueCount = 1,
	    .pClearValues = &color,
	};
	vkCmdBeginRenderPass(frame->cmd_buffer, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(frame->cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline.pipeline.value());
	VkViewport viewport {0.0f, 0.0f, (float)vbc->swapchain_extent.width, (float)vbc->swapchain_extent.height};
	vkCmdSetViewport(frame->cmd_buffer, 0, 1, &viewport);
	VkRect2D scissor {{0,0}, vbc->swapchain_extent};
	vkCmdSetScissor(frame->cmd_buffer, 0, 1, &scissor);

	vkCmdPushConstants(frame->cmd_buffer, graphics_pipeline.layout.value(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &push_constants);
	vkCmdDraw(frame->cmd_buffer, 3, 1, 0, 0);
	vkCmdEndRenderPass(frame->cmd_buffer);

	vb::render::end_command_buffer(frame->cmd_buffer);

	vb::render::submit_queue(vbc->queues_info.graphics_queue, frame->render_fence,
		{frame->cmd_buffer}, {frame->image_available_semaphore}, {frame->finish_render_semaphore}, 
		{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT});

	vb::render::present_queue(vbc->queues_info.graphics_queue, &vbc->swapchain,
		{frame->finish_render_semaphore}, &image_index);

	vbc->frame_index++;
    }
    vkDeviceWaitIdle(vbc->device);

    vkDestroyRenderPass(vbc->device, render_pass, nullptr);
    graphics_pipeline.clean();
    vertex_buffer.clean();
}
