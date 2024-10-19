#include <SDL3/SDL_events.h>
#include <memory>
#include <vb.h>
#include <vulkan/vulkan_core.h>

int main(int argc, char** argv) {
    auto info = vb::Context::Info {
	.title = "vbc",
	.width = 800,
	.height = 600,
    };
    auto vbc = std::make_unique<vb::Context>(info);
    auto graphics_pipeline = vb::builder::GraphicsPipeline{vbc.get()};
    graphics_pipeline.add_shader("../shaders/triangle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    graphics_pipeline.add_shader("../shaders/triangle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

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
    VB_ASSERT(graphics_pipeline.pipeline);

    VkFramebuffer framebuffers[vbc->swapchain_image_views.size()];
    for(size_t i = 0; i < vbc->swapchain_image_views.size(); i++) {
	VkImageView attachments[1] = {vbc->swapchain_image_views[i]};
	VkFramebufferCreateInfo framebuffer_info {
	    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
    	    .renderPass = render_pass,
    	    .attachmentCount = 1,
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

	auto frame = vbc->get_current_frame();
	auto image = vbc->wait_on_image_reset_fence(frame);
	if(!image.has_value()) continue;
	uint32_t image_index = image.value();

 	vkResetCommandBuffer(frame->cmd_buffer, 0);
 	VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
 	VB_ASSERT(vkBeginCommandBuffer(frame->cmd_buffer, &begin) == VK_SUCCESS);

	VkClearValue color {{{0.0f, 0.0f, 0.0f, 1.0f}}};
	VB_ASSERT(framebuffers[image_index] != VK_NULL_HANDLE);
	VkRenderPassBeginInfo render_begin {
	    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
	    .renderPass = render_pass,
	    .framebuffer = framebuffers[image_index],
	    .renderArea = {{0,0}, vbc->swapchain_extent},
	    .clearValueCount = 1,
	    .pClearValues = &color,
	};
	vkCmdBeginRenderPass(frame->cmd_buffer, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(frame->cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline.pipeline);
	VkViewport viewport {0.0f, 0.0f, (float)vbc->swapchain_extent.width, (float)vbc->swapchain_extent.height};
	vkCmdSetViewport(frame->cmd_buffer, 0, 1, &viewport);
	VkRect2D scissor {{0,0}, vbc->swapchain_extent};
	vkCmdSetScissor(frame->cmd_buffer, 0, 1, &scissor);
	vkCmdDraw(frame->cmd_buffer, 3, 1, 0, 0);
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
    for(size_t i=0; i<vbc->swapchain_image_views.size(); i++) vkDestroyFramebuffer(vbc->device, framebuffers[i], nullptr);
}
