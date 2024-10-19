#include <vbc.h>
#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
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

struct DispatchData {
    VBBuffer* staging_buffer;
    size_t vertices_size;
    VBBuffer* vertex_buffer;
    size_t indices_size;
    VBBuffer* index_buffer;
};

void dispatch(VkCommandBuffer cmd, void* data) {
    DispatchData* d = (DispatchData*)data;
    VkBufferCopy copy = { .size = d->vertices_size };
    vkCmdCopyBuffer(cmd, d->staging_buffer->buffer, d->vertex_buffer->buffer, 1, &copy);
    VkBufferCopy copy2 = { .srcOffset = d->vertices_size, .size = d->indices_size };
    vkCmdCopyBuffer(cmd, d->staging_buffer->buffer, d->index_buffer->buffer, 1, &copy2);
}

struct Rectangle {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VBBuffer* vertex_buffer;
    VBBuffer* index_buffer;
    VkDeviceAddress vertex_buffer_address;

    Rectangle(VBContext* context, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices): vertices{vertices}, indices{indices} {
	    const size_t vertices_size = sizeof(Vertex) * vertices.size();
	    vertex_buffer = vb_new_buffer(context, vertices_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
		    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VMA_MEMORY_USAGE_GPU_ONLY);
	    VB_ASSERT(vertex_buffer);
	    VkBufferDeviceAddressInfo address = {
    	        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    	        .buffer = vertex_buffer->buffer,
    	    };
    	    vertex_buffer_address = vkGetBufferDeviceAddress(context->device, &address);
	    const size_t indices_size = sizeof(uint32_t) * indices.size();
	    index_buffer = vb_new_buffer(context, indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT
		    | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	    VB_ASSERT(index_buffer);

	    auto staging_buffer = vb_new_buffer(context, vertices_size + indices_size,
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	    VB_ASSERT(staging_buffer);
    	    memcpy(staging_buffer->info.pMappedData, vertices.data(), vertices_size);
    	    memcpy((char*)(staging_buffer->info.pMappedData)+vertices_size, indices.data(), indices_size);
	    DispatchData data = {staging_buffer, vertices_size, vertex_buffer, indices_size, index_buffer};
	    vb_dispatch_command(context, dispatch, &data);
	    vb_free_buffer(staging_buffer);
    }
};

int main(int argc, char** argv) {
    VkPhysicalDeviceFeatures vk10 { 
	.samplerAnisotropy = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features vk12 {
	.imagelessFramebuffer = VK_TRUE,
	.separateDepthStencilLayouts = VK_TRUE,
	.bufferDeviceAddress = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features vk13 {
	.synchronization2 = VK_TRUE,
    };
    VBContextInfo info {
	.title = "vbc",
	.width = 800,
	.height = 600,
	.vk10features = vk10,
	.vk12features = vk12,
	.vk13features = vk13,
    };
    auto vbc = vb_new_context(&info);

    int tw, th, tc;
    stbi_uc* data = stbi_load("../textures/texture.jpg", &tw, &th, &tc, STBI_rgb_alpha);
    VB_ASSERT(data);
    VkExtent3D size = {(uint32_t)tw,(uint32_t)th,1};
    auto texture = vb_new_image_from_data(vbc, data, size, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);
    VB_ASSERT(texture);

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
    Rectangle rectangle {vbc, vertices, indices};
    const std::vector<Vertex> vertices2 = {
    	{{-0.5f, -0.5f, -0.5f}, 1.0f, {1.0f, 1.0f, 1.0f}, 0.0f, {1.0f, 0.0f, 0.0f, 1.0f}},
	{{0.5f,  -0.5f, -0.5f}, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f, {0.0f, 1.0f, 0.0f, 1.0f}},
    	{{0.5f,  0.5f,  -0.5f}, 0.0f, {1.0f, 1.0f, 1.0f}, 1.0f, {0.0f, 0.0f, 1.0f, 1.0f}},
    	{{-0.5f, 0.5f,  -0.5f}, 1.0f, {1.0f, 1.0f, 1.0f}, 1.0f, {1.0f, 1.0f, 0.0f, 1.0f}}
    };
    Rectangle rectangle2 {vbc, vertices2, indices};

    // VkDescriptorSetLayoutBinding binding {
    //     .binding = 0,
    //     .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    //     .descriptorCount = 1,
    //     .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    // };
    // VkDescriptorSetLayoutCreateInfo layout_info {
    //     .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    //     .bindingCount = 1,
    //     .pBindings = &binding,
    // };
    // VkDescriptorSetLayout layout;
    // VB_ASSERT(vkCreateDescriptorSetLayout(vbc->device, &layout_info, nullptr, &layout) == VK_SUCCESS);
    // auto descriptor_builder = vb::builder::Descriptor(vbc.get());
    // vb::builder::Descriptor::Ratio sizes[1] {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}};
    // descriptor_builder.create(sizes);
    // auto set = descriptor_builder.allocate(layout);
    // VB_ASSERT(set.has_value());

    // VkDescriptorImageInfo image_info {
    //     .sampler = sampler,
    //     .imageView = texture.image_view.value(),
    //     .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    // };
    // VkWriteDescriptorSet descriptor_write {
    //     .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    //     .dstSet = set.value(),
    //     .dstBinding = 0,
    //     .dstArrayElement = 0,
    //     .descriptorCount = 1,
    //     .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    //     .pImageInfo = &image_info,
    // };
    // vkUpdateDescriptorSets(vbc->device, 1, &descriptor_write, 0, nullptr);

    VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };
    VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
    };
    VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
	.depthTestEnable = VK_TRUE,
	.depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    auto vert_sh = vb_create_shader_module(vbc->device, "../shaders/full_vert.vert.spv");
    VB_ASSERT(vert_sh);
    auto frag_sh = vb_create_shader_module(vbc->device, "../shaders/triangle.frag.spv");
    VB_ASSERT(frag_sh);

    VkPipelineShaderStageCreateInfo shaders[2] = {
	{
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	    .stage = VK_SHADER_STAGE_VERTEX_BIT,
	    .module = vert_sh,
	    .pName = "main",
	},
	{
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
	    .module = frag_sh,
	    .pName = "main",
	},
    };

    VkPushConstantRange range = {VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants)};

    auto depth_image = vb_new_image(vbc, {vbc->swapchain_extent.width, vbc->swapchain_extent.height, 1}, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, false);
    VB_ASSERT(depth_image);

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
	.pushConstantRangeCount = 1,
	.pPushConstantRanges = &range,
    };
    VkPipelineLayout layout;
    VB_ASSERT(vkCreatePipelineLayout(vbc->device, &pipeline_layout, nullptr, &layout) != VK_SUCCESS);
    VkGraphicsPipelineCreateInfo pipeline_info = {
       .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
       .stageCount = 2,
       .pStages = shaders,
       .pVertexInputState = &vertex_input,
       .pInputAssemblyState = &input_assembly,
       //.pTessellationState = tessellation,
       .pViewportState = &viewport,
       .pRasterizationState = &rasterization,
       .pMultisampleState = &multisample,
       .pDepthStencilState = &depth_stencil,
       .pColorBlendState = &color_blend,
       .pDynamicState = &dynamic_state,
       .layout = layout,
       .renderPass = render_pass,
       .subpass = 0,
    };
    VkPipeline pipeline;
    VB_ASSERT(vkCreateGraphicsPipelines(vbc->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline) != VK_SUCCESS);

    VkFramebufferAttachmentImageInfo framebuffer_color_attachment_info {
	.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO,
	.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
	.width = vbc->swapchain_extent.width,
	.height = vbc->swapchain_extent.height,
	.layerCount = 1,
	.viewFormatCount = 1,
	.pViewFormats = &vbc->swapchain_format,
    };

    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;
    VkFramebufferAttachmentImageInfo framebuffer_depth_attachment_info {
	.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO,
	.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
	.width = vbc->swapchain_extent.width,
	.height = vbc->swapchain_extent.height,
	.layerCount = 1,
	.viewFormatCount = 1,
	.pViewFormats = &depth_format,
    };

    VkFramebufferAttachmentImageInfo framebuffer_attachments[2] = {framebuffer_color_attachment_info, framebuffer_depth_attachment_info};

    VkFramebufferAttachmentsCreateInfo framebuffer_attachments_info {
	.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO,
	.attachmentImageInfoCount = 2,
	.pAttachmentImageInfos = framebuffer_attachments,
    };

    VkFramebufferCreateInfo framebuffer_info {
    	.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
	.pNext = &framebuffer_attachments_info,
	.flags = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT,
	.renderPass = render_pass,
	.attachmentCount = 2,
	.width = vbc->swapchain_extent.width,
	.height = vbc->swapchain_extent.height,
	.layers = 1,
    };
    VkFramebuffer framebuffer;
    VB_ASSERT(vkCreateFramebuffer(vbc->device, &framebuffer_info, nullptr, &framebuffer) == VK_SUCCESS);
    // vbc->create_swapchain_framebuffers(render_pass, {depth_image.image_view.value()});

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
	    vb_recreate_swapchain(vbc);
	    vb_free_image(depth_image);
	    vkDestroyFramebuffer(vbc->device, framebuffer, nullptr);
	    depth_image = vb_new_image(vbc, {vbc->swapchain_extent.width, vbc->swapchain_extent.height, 1}, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, false);
	    VkFramebufferAttachmentImageInfo framebuffer_color_attachment_info {
    	        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO,
    	        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    	        .width = vbc->swapchain_extent.width,
    	        .height = vbc->swapchain_extent.height,
    	        .layerCount = 1,
    	        .viewFormatCount = 1,
    	        .pViewFormats = &vbc->swapchain_format,
    	    };

    	    VkFramebufferAttachmentImageInfo framebuffer_depth_attachment_info {
    	        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO,
    	        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    	        .width = vbc->swapchain_extent.width,
    	        .height = vbc->swapchain_extent.height,
    	        .layerCount = 1,
    	        .viewFormatCount = 1,
    	        .pViewFormats = &depth_format,
    	    };

    	    VkFramebufferAttachmentImageInfo framebuffer_attachments[2] = {framebuffer_color_attachment_info, framebuffer_depth_attachment_info};

    	    VkFramebufferAttachmentsCreateInfo framebuffer_attachments_info {
    	        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO,
    	        .attachmentImageInfoCount = 2,
    	        .pAttachmentImageInfos = framebuffer_attachments,
    	    };
	    VkFramebufferCreateInfo framebuffer_info {
    	    	.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
    	        .pNext = &framebuffer_attachments_info,
    	        .flags = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT,
    	        .renderPass = render_pass,
    	        .attachmentCount = 2,
    	        .width = vbc->swapchain_extent.width,
    	        .height = vbc->swapchain_extent.height,
    	        .layers = 1,
    	    };
    	    VB_ASSERT(vkCreateFramebuffer(vbc->device, &framebuffer_info, nullptr, &framebuffer) == VK_SUCCESS);
	}

	auto frame = &vbc->frames[vbc->frame_index % VB_MAX_FRAMES];
	uint32_t image_index = 0;
	vkWaitForFences(vbc->device, 1, &frame->render_fence, VK_TRUE, UINT64_MAX);
	VkResult result = vkAcquireNextImageKHR(vbc->device, vbc->swapchain, UINT64_MAX, frame->image_available_semaphore, VK_NULL_HANDLE, &image_index);
	if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
	    vbc->resize = true;
	    continue;
	}
	vkResetFences(vbc->device, 1, &frame->render_fence);

 	vkResetCommandBuffer(frame->cmd_buffer, 0);
 	VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
 	VB_ASSERT(vkBeginCommandBuffer(frame->cmd_buffer, &begin) == VK_SUCCESS);

	VkClearValue color[2] {
	    {.color = {0.0f, 0.0f, 0.0f, 1.0f}},
	    {.depthStencil = {1.0f, 0}},
	};
	VkImageView views[2] = {vbc->swapchain_image_views[image_index], depth_image->image_view};
	VkRenderPassAttachmentBeginInfo render_begin_attachments {
	    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO,
	    .attachmentCount = 2,
	    .pAttachments = views,
	};
	VkRenderPassBeginInfo render_begin {
	    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
	    .pNext = &render_begin_attachments,
	    .renderPass = render_pass,
	    .framebuffer = framebuffer,
	    .renderArea = {{0,0}, vbc->swapchain_extent},
	    .clearValueCount = 2,
	    .pClearValues = color,
	};
	vkCmdBeginRenderPass(frame->cmd_buffer, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(frame->cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkViewport viewport {0.0f, 0.0f, (float)vbc->swapchain_extent.width, (float)vbc->swapchain_extent.height};
	vkCmdSetViewport(frame->cmd_buffer, 0, 1, &viewport);
	VkRect2D scissor {{0,0}, vbc->swapchain_extent};
	vkCmdSetScissor(frame->cmd_buffer, 0, 1, &scissor);

	// vkCmdBindDescriptorSets(frame->cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline.layout.value(), 0, 1, &set.value(), 0, nullptr);
	// glm::mat4 model = glm::mat4(1);
	// model = glm::rotate(model, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)vbc->swapchain_extent.width/(float)vbc->swapchain_extent.height, 0.1f, 100.0f);
	proj[1][1] *= -1;
	auto push_constants = PushConstants{proj * view, rectangle.vertex_buffer_address};
	vkCmdPushConstants(frame->cmd_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &push_constants);
	vkCmdBindIndexBuffer(frame->cmd_buffer, rectangle.index_buffer->buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(frame->cmd_buffer, (uint32_t)indices.size(), 2, 0, 0, 0);
	push_constants = PushConstants{proj * view, rectangle2.vertex_buffer_address};
	vkCmdPushConstants(frame->cmd_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &push_constants);
	vkCmdBindIndexBuffer(frame->cmd_buffer, rectangle2.index_buffer->buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(frame->cmd_buffer, (uint32_t)indices.size(), 2, 0, 0, 0);
	vkCmdEndRenderPass(frame->cmd_buffer);

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

    vkDestroyRenderPass(vbc->device, render_pass, nullptr);
    vkDestroyPipeline(vbc->device, pipeline, nullptr);
    vkDestroyPipelineLayout(vbc->device, layout, nullptr);
    vb_free_buffer(rectangle.vertex_buffer);
    vb_free_buffer(rectangle.index_buffer);
    vb_free_buffer(rectangle2.vertex_buffer);
    vb_free_buffer(rectangle2.index_buffer);
    vb_free_image(texture);
    vkDestroySampler(vbc->device, sampler, nullptr);
    //vkDestroyDescriptorSetLayout(vbc->device, layout, nullptr);
    //descriptor_builder.clean();
    vb_free_image(depth_image);
    vkDestroyFramebuffer(vbc->device, framebuffer, nullptr);
}
