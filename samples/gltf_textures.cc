#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <stb/stb_image.h>
#include <format>
#include <vb.h>
#include <imgui.h>
#include <chrono>
#include <algorithm>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include <filesystem>

struct GLTF {
    vb::Context* ctx;

    struct Vertex {
	    glm::vec3 position;
	    float uv_x;
	    glm::vec3 normal;
	    float uv_y;
	    glm::vec4 color;
	};

    struct Primitive {
        uint32_t first_index;
        uint32_t index_count;
        uint32_t material_index;
    };

    struct Mesh {
        std::vector<Primitive> primitives;
    };

    struct Node;
    struct Node {
        Node* parent;
        std::vector<Node*> children;
        Mesh mesh;
        glm::mat4 matrix;
        void clean() { for(auto& child: children) delete child; }
    };

    struct Material {
        glm::vec4 base_color_factor {1.0f};
        uint32_t base_color_tex_index;
        uint32_t normal_tex_index;
        float alpha_cutoff;
        bool double_sided {false};
    };

    struct Image {
        vb::Image texture;
        VkDescriptorSet set;
    };

    struct Texture {
        uint32_t index;
    };

    VkDescriptorSetLayout descriptor_set_layout;
    vb::DescriptorAllocator descriptor_pool;

    VkSampler sampler;

    VkDeviceAddress vertices_addr;
    vb::Buffer vertices;
    vb::Buffer indices;

    std::vector<Image> images;
    std::vector<Texture> textures;
    std::vector<Material> materials;
    std::vector<Node*> nodes;

    GLTF(vb::Context* context): ctx{context}, descriptor_pool{context},
        vertices{context}, indices{context} {}
    void load_node(const fastgltf::Node& node_in, GLTF::Node* parent, 
    	const fastgltf::Asset& asset, std::vector<Vertex>& vertex_vec,
    	std::vector<uint32_t>& index_vec);
    void load_material(const fastgltf::Asset& asset, fastgltf::Material& material);
    void load_image(const fastgltf::Asset& asset, fastgltf::Image& image,
    	const std::filesystem::path& parent_path, vb::CommandPool& pool);
    void setup_descriptors();
    void setup_sampler();
    void load(const std::filesystem::path& path, vb::CommandPool& pool,
        VkCommandBuffer cmd);
    void clean();
};

struct PushConstants {
    glm::mat4 render_matrix;
    VkDeviceAddress vertex_buffer;
};

struct Camera {
    glm::vec3 velocity {0.0f, 0.0f, 0.0f};
    glm::vec3 position {0.0f, 0.0f, 5.0f};
    float rot_speed {100.0f};
    float move_speed {0.1f};
    float pitch {0.0f};
    float yaw {0.0f};
    bool lock {false};

    glm::mat4 get_view();
    glm::mat4 get_rotation();
    void handle_event(SDL_Event& event);
    void update();
};

struct App {
    vb::ContextInfo vb_context_info = {
	.title = "vkgfxrenderer",
	.width = 1280,
	.height = 720,
	.enable_debug = true,
	.vk10features = {
    	    .samplerAnisotropy = VK_TRUE,
        },
	.vk11features = {},
	.vk12features = {
    	    .descriptorIndexing = VK_TRUE,
	    .bufferDeviceAddress = VK_TRUE,
        },
    	.vk13features = {
	    .dynamicRendering = VK_TRUE,
        },
        .present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR,
    };
    vb::Context vbc {vb_context_info};

    struct {
        uint64_t fps;
        float frametime;
        uint64_t triangles;
        uint64_t drawcalls;
        float update_time;
        float draw_time;
    } stats;

    bool running {true};
    bool resize {false};

    vb::CommandPool frames_cmdpool;
    struct Frame {
	    VkCommandBuffer cmd;
	    VkSemaphore image_available;
	    VkSemaphore finish_render;
	    VkFence render;
	};
    std::vector<Frame> frames;
    uint8_t frame_index;

    VkDescriptorPool imgui_descriptor_pool;

    vb::CommandPool cmdpool;
    VkCommandBuffer global_cmd_buffer;

    float aspect_ratio {0.0f};
    VkExtent2D render_extent;
    vb::Image render_target;
    vb::Image depth_target;

    vb::GraphicsPipeline gfx_pipeline;
    GLTF mesh;

    struct {
        glm::mat4 projection;
        glm::mat4 view;
    } scene_data;
    Camera camera;

    App();
    ~App();

    void init_frames();
    void init_imgui();

    void create_target_images();
    void destroy_target_images();
    void init_pipelines();
    void load_mesh();

    void recreate_targets();

    VkImageLayout render_imgui(VkCommandBuffer cmd, VkImageLayout input_layout, uint32_t index);
    void imgui_interface();

    void run();
    VkImageLayout render(VkCommandBuffer cmd, VkImageLayout input_layout, uint32_t index);

    std::string save_screenshot(VkImage image);
};

    App::App(): frames_cmdpool{&vbc}, cmdpool{&vbc}, gfx_pipeline{&vbc},
    mesh{&vbc}, render_target{&vbc}, depth_target{&vbc} {
	init_frames();
	init_imgui();
	vbc.set_resize_callback([&]() {recreate_targets();});
	cmdpool.create(vbc.queues_info.graphics_queue, vbc.queues_info.graphics_index);
	VB_ASSERT(cmdpool.all_valid());
	global_cmd_buffer = cmdpool.allocate();

	create_target_images();
	load_mesh();
	init_pipelines();
	SDL_SetWindowRelativeMouseMode(vbc.window, true);
    }

    App::~App() {
	mesh.clean();
	destroy_target_images();
	gfx_pipeline.clean();
	cmdpool.clean();
	ImGui_ImplVulkan_Shutdown();
	vkDestroyDescriptorPool(vbc.device, imgui_descriptor_pool, nullptr);
	for(auto& frame: frames) {
	    vkDestroyFence(vbc.device, frame.render, nullptr);
	    vkDestroySemaphore(vbc.device, frame.finish_render, nullptr);
	    vkDestroySemaphore(vbc.device, frame.image_available, nullptr);
	}
	frames_cmdpool.clean();
    }

    void App::init_frames() {
	frames_cmdpool.create(vbc.queues_info.graphics_queue, vbc.queues_info.graphics_index);
	VB_ASSERT(frames_cmdpool.all_valid());
	frames.resize(vbc.swapchain_image_views.size());
	for(auto& frame: frames) {
	    frame.cmd = frames_cmdpool.allocate();
	    VB_ASSERT(frame.cmd);
	    frame.finish_render = vb::create_semaphore(vbc.device);
	    VB_ASSERT(frame.finish_render);
	    frame.image_available = vb::create_semaphore(vbc.device);
	    VB_ASSERT(frame.image_available);
	    frame.render = vb::create_fence(vbc.device, VK_FENCE_CREATE_SIGNALED_BIT);
	    VB_ASSERT(frame.render);
	}
    }

    void App::create_target_images() {
	aspect_ratio = (float)vbc.info.height / (float)vbc.info.width;
	render_extent.width = std::min(vbc.info.width, vbc.swapchain_extent.width);
	render_extent.height = std::min(vbc.info.width,
		(uint32_t)(aspect_ratio * (float)vbc.swapchain_extent.width));

	render_target.create({render_extent.width, render_extent.height, 1},
		VK_FORMAT_B8G8R8A8_SRGB, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	VB_ASSERT(render_target.all_valid());

	depth_target.create({vbc.swapchain_extent.width, vbc.swapchain_extent.height, 1},
		VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
	VB_ASSERT(depth_target.all_valid());
    }

    void App::destroy_target_images() {
	render_target.clean();
	depth_target.clean();
    }

    void App::load_mesh() {
	mesh.load("/home/mar/CG/glTF-Sample-Assets/Models/Sponza/glTF/Sponza.gltf",
		cmdpool, global_cmd_buffer);
	VB_ASSERT(mesh.vertices.all_valid()&&mesh.indices.all_valid());
    }

    void App::init_pipelines() {
	gfx_pipeline.set_front_face(VK_FRONT_FACE_COUNTER_CLOCKWISE);
	gfx_pipeline.color_blend_attachment.blendEnable = VK_TRUE;
	gfx_pipeline.color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	gfx_pipeline.color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	gfx_pipeline.color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
	gfx_pipeline.color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	gfx_pipeline.color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	gfx_pipeline.color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
	gfx_pipeline.enable_depth_test();
	gfx_pipeline.set_depth_comparison(VK_COMPARE_OP_GREATER_OR_EQUAL);

	gfx_pipeline.add_shader("../shaders/full_vert.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
	gfx_pipeline.add_shader("../shaders/basictex.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
	gfx_pipeline.add_push_constant(sizeof(PushConstants), VK_SHADER_STAGE_VERTEX_BIT);
	gfx_pipeline.add_descriptor_set_layout(mesh.descriptor_set_layout);
	VkPipelineRenderingCreateInfo info = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
	    .colorAttachmentCount = 1,
	    .pColorAttachmentFormats = &vbc.swapchain_format,
	    .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
	};
	gfx_pipeline.create(&info, 0);
	VB_ASSERT(gfx_pipeline.all_valid());
    }

    void App::recreate_targets() {
	vbc.recreate_swapchain([&](uint32_t w, uint32_t h) {
	    destroy_target_images();
	});
	create_target_images();
	resize = false;
    }

    VkImageLayout App::render(VkCommandBuffer cmd, VkImageLayout input_layout, uint32_t index) {
	stats.drawcalls = 0;
	stats.triangles = 0;
	auto start = std::chrono::high_resolution_clock::now();

	vb::transition_image(cmd, render_target.image, VK_IMAGE_LAYOUT_UNDEFINED, 
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	vb::transition_image(cmd, depth_target.image, VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
	VkClearValue color[2] = {
	    {.color = {0.0f, 0.0f, 0.0f, 1.0f}},
	    {.depthStencil = {0.0f, 0}},
	};
	VkRenderingAttachmentInfo color_attach = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	    .imageView = render_target.image_view,
	    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	    .clearValue = color[0],
	};
	VkRenderingAttachmentInfo depth_attach = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	    .imageView = depth_target.image_view,
	    .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
	    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	    .clearValue = color[1],
	};
	VkRenderingInfo rendering = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
	    .renderArea = {{0,0}, render_extent},
	    .layerCount = 1,
	    .colorAttachmentCount = 1,
	    .pColorAttachments = &color_attach,
	    .pDepthAttachment = &depth_attach,
	};
	vkCmdBeginRendering(cmd, &rendering);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfx_pipeline.pipeline);
	VkViewport viewport = {0.0f, 0.0f, (float)render_extent.width,
	    (float)render_extent.height, 0.0f, 1.0f};
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	VkRect2D scissor {{0,0}, render_extent};
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBindIndexBuffer(cmd, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);

	for(const auto& node: mesh.nodes) {
	    if(node->mesh.primitives.size() > 0) {
		auto node_matrix = node->matrix;
		auto parent = node->parent;
		while(parent) {
		    node_matrix = parent->matrix * node_matrix;
		    parent = parent->parent;
		}
		auto push_constants = PushConstants{
		    scene_data.projection * scene_data.view * node_matrix,
		    mesh.vertices_addr};
		vkCmdPushConstants(cmd, gfx_pipeline.layout,
		    VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &push_constants);
		for(auto& primitive: node->mesh.primitives) {
		    if(primitive.index_count > 0) {
			auto texture = mesh.textures[
			    mesh.materials[primitive.material_index].base_color_tex_index];
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			    gfx_pipeline.layout, 0, 1, &mesh.images[texture.index].set,
			    0, nullptr);
			vkCmdDrawIndexed(cmd, primitive.index_count, 1, 
		    	    primitive.first_index, 0, 0);
			stats.drawcalls++;
			stats.triangles += primitive.index_count/3;
		    }
		}
	    }
	}

	vkCmdEndRendering(cmd);
	auto end = std::chrono::high_resolution_clock::now();
    	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	stats.draw_time = elapsed.count() / 1000.0f;

	vb::transition_image(cmd, render_target.image,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	vb::transition_image(cmd, vbc.swapchain_images[index], input_layout,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	vb::blit_image(cmd, render_target.image, vbc.swapchain_images[index],
		{render_extent.width, render_extent.height, 1},
		{vbc.swapchain_extent.width, vbc.swapchain_extent.height, 1});

	return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    }

    void App::run() {
        SDL_Event event;
        while(running) {
            while(SDL_PollEvent(&event) != 0) {
                switch(event.type) {
		    case SDL_EVENT_QUIT:
            	        running = false;
            	        break;
            	    case SDL_EVENT_WINDOW_RESIZED: case SDL_EVENT_WINDOW_MAXIMIZED:
            	    case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
		    case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
            	        resize = true;
            	        break;
            	    case SDL_EVENT_WINDOW_HIDDEN: case SDL_EVENT_WINDOW_MINIMIZED:
		    case SDL_EVENT_WINDOW_OCCLUDED:
            	        SDL_WaitEvent(&event);
            	        break;
		    case SDL_EVENT_KEY_UP:
			if(event.key.key == SDLK_Q) {
			    camera.lock = camera.lock ? false : true;
			    SDL_SetWindowRelativeMouseMode(vbc.window, camera.lock ? false : true);
			}
			break;
                }
		if(!camera.lock) camera.handle_event(event);
		ImGui_ImplSDL3_ProcessEvent(&event);
            }

	    if(resize) recreate_targets();

	    auto start = std::chrono::high_resolution_clock::now();

	    if(!camera.lock) camera.update();
	    scene_data.view = camera.get_view();
	    glm::mat4 proj = glm::perspective(glm::radians(45.0f),
		(float)vbc.swapchain_extent.width/(float)vbc.swapchain_extent.height,
		1000.0f, 0.1f);
	    proj[1][1] *= -1;
	    scene_data.projection = proj;


	    ImGui_ImplVulkan_NewFrame();
	    ImGui_ImplSDL3_NewFrame();
	    ImGui::NewFrame();
	    imgui_interface();
	    ImGui::Render();

	    auto frame = &frames[frame_index%frames.size()];
	    VB_ASSERT(vkWaitForFences(vbc.device, 1, &frame->render, VK_TRUE, UINT64_MAX)
		    == VK_SUCCESS);
	    auto next = vbc.acquire_next_image(frame->image_available);
	    if(!next.has_value()) continue;
	    uint32_t index = next.value();
	    vkResetFences(vbc.device, 1, &frame->render);
	    vkResetCommandBuffer(frame->cmd, 0);
	    VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	    };
	    VB_ASSERT(vkBeginCommandBuffer(frame->cmd, &begin) == VK_SUCCESS);

	    auto layout = VK_IMAGE_LAYOUT_UNDEFINED;
	    layout = render(frame->cmd, layout, index);
	    layout = render_imgui(frame->cmd, layout, index);

	    vb::transition_image(frame->cmd, vbc.swapchain_images[index],
		    layout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	    VB_ASSERT(vkEndCommandBuffer(frame->cmd) == VK_SUCCESS);

	    VkPipelineStageFlags mask[1] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
	    VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &frame->image_available,
		.pWaitDstStageMask = mask,
		.commandBufferCount = 1,
		.pCommandBuffers = &frame->cmd,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &frame->finish_render,
	    };
	    VB_ASSERT(vkQueueSubmit(vbc.queues_info.graphics_queue, 1, &submit, frame->render)
		    == VK_SUCCESS);

	    VkPresentInfoKHR present = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &frame->finish_render,
		.swapchainCount = 1,
		.pSwapchains = &vbc.swapchain,
		.pImageIndices = &index,
	    };
	    vkQueuePresentKHR(vbc.queues_info.graphics_queue, &present);
	    frame_index++;

	    auto end = std::chrono::high_resolution_clock::now();
	    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	    stats.frametime = elapsed.count() / 1000.0f;
	    stats.fps = (float)(1.0f / stats.frametime) * 1000.0f;
        }
        vkDeviceWaitIdle(vbc.device);
    }

    glm::mat4 Camera::get_view() {
	auto trans = glm::translate(glm::mat4(1.0f), position);
	auto rot = get_rotation();
	return glm::inverse(trans * rot);
    }

    glm::mat4 Camera::get_rotation() {
	auto pitch_rot = glm::angleAxis(pitch, glm::vec3(1.0f, 0.0f, 0.0f));
	auto yaw_rot = glm::angleAxis(yaw, glm::vec3(0.0f, -1.0f, 0.0f));
	return glm::mat4(yaw_rot) * glm::mat4(pitch_rot);
    }

    void Camera::handle_event(SDL_Event& event){ 
	if(event.type == SDL_EVENT_KEY_DOWN) {
	    switch(event.key.key) {
		case SDLK_W: velocity.z = -1; break;
		case SDLK_S: velocity.z = 1; break;
		case SDLK_A: velocity.x = -1; break;
		case SDLK_D: velocity.x = 1; break;
	    }
	}
	if(event.type == SDL_EVENT_KEY_UP) {
	    switch(event.key.key) {
		case SDLK_W: velocity.z = 0; break;
		case SDLK_S: velocity.z = 0; break;
		case SDLK_A: velocity.x = 0; break;
		case SDLK_D: velocity.x = 0; break;
	    }
	}
	if(event.type == SDL_EVENT_MOUSE_MOTION) {
	    yaw += (float)event.motion.xrel / rot_speed;
	    pitch -= (float)event.motion.yrel / rot_speed;
	    if(pitch < -89.0f) pitch = -89.0f;
	    if(pitch > 89.0f) pitch = 89.0f;
	}
    }

    void Camera::update() {
	auto rot = get_rotation();
	position += glm::vec3(rot * glm::vec4(velocity * move_speed, 0.0f));
    }

    void GLTF::clean() {
	descriptor_pool.clean();
	vkDestroyDescriptorSetLayout(ctx->device, descriptor_set_layout, nullptr);
	vkDestroySampler(ctx->device, sampler, nullptr);
	vertices.clean();
	indices.clean();
	for(auto& image: images) image.texture.clean();
	for(auto& node: nodes) {
	    node->clean();
	    delete node;
	}
    }

    void GLTF::load(const std::filesystem::path& path, vb::CommandPool& pool,
	    VkCommandBuffer cmd) {
	vb::log(std::format("Loading {}...", path.string()));
	fastgltf::Parser parser;
	auto data = fastgltf::GltfDataBuffer::FromPath(path);
	VB_ASSERT(data.error() == fastgltf::Error::None);
	auto options = fastgltf::Options::LoadExternalBuffers
	    | fastgltf::Options::GenerateMeshIndices
	    | fastgltf::Options::DecomposeNodeMatrices;
	auto parent_path = path.parent_path();
	auto asset = parser.loadGltf(data.get(), parent_path, options);
	VB_ASSERT(asset.error() == fastgltf::Error::None);
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	// IMAGES
	for(auto& image: asset->images) 
	    load_image(asset.get(), image, parent_path, pool);
	// TEXTURES
	for(auto& texture: asset->textures) {
	    if(texture.imageIndex.has_value())
		textures.push_back({(uint32_t)texture.imageIndex.value()});
	}
	// MATERIALS
	for(auto& material: asset->materials) {
	    load_material(asset.get(), material);
	}
	// NODES
	for(auto& node: asset->nodes) {
	    load_node(node, nullptr, asset.get(), vertices, indices);
	}

	vb::log("All GLTF data loaded");
	std::vector<vb::DescriptorAllocator::Ratio> sizes = {
	    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (float)images.size()},
	};
	VB_ASSERT(images.size() > 0);
	descriptor_pool.create(sizes, images.size()+1);
	setup_sampler();
	setup_descriptors();

	vb::log("Creating buffers...");
	size_t vertices_size = vertices.size() * sizeof(Vertex);
	this->vertices.create(vertices_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);
	VB_ASSERT(this->vertices.all_valid());
	VkBufferDeviceAddressInfo addr_info = {
	    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
	    .buffer = this->vertices.buffer,
	};
	vertices_addr = vkGetBufferDeviceAddress(ctx->device, &addr_info);
	size_t indices_size = indices.size() * sizeof(uint32_t);
	this->indices.create(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT
		| VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	VB_ASSERT(this->indices.all_valid());
	vb::Buffer staging_buffer {ctx};
	staging_buffer.create(vertices_size + indices_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_ONLY);
	VB_ASSERT(staging_buffer.all_valid());
	char* buf_data = (char*)staging_buffer.info.pMappedData;
	memcpy(buf_data, vertices.data(), vertices_size);
	memcpy(buf_data + vertices_size, indices.data(), indices_size);
	vb::log("Copying data to buffers...");
	pool.submit_command_buffer_to_queue(cmd, [&](VkCommandBuffer cmd) {
	    VkBufferCopy copy = { .size = vertices_size };
	    vkCmdCopyBuffer(cmd, staging_buffer.buffer, this->vertices.buffer, 1, &copy);
	    VkBufferCopy copy2 = {
		.srcOffset = vertices_size,
		.size = indices_size,
	    };
	    vkCmdCopyBuffer(cmd, staging_buffer.buffer, this->indices.buffer, 1, &copy2);
	});
	staging_buffer.clean();
	vb::log("GLTF object created");
    }

    void GLTF::setup_descriptors() {
	VkDescriptorSetLayoutBinding binding = {
	    .binding = 0,
	    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	    .descriptorCount = 1,
	    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	VkDescriptorSetLayoutCreateInfo layout_info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    .bindingCount = 1,
	    .pBindings = &binding,
	};
	VB_ASSERT(vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr,
		    &descriptor_set_layout) == VK_SUCCESS);
	for(auto& image: images) {
	    image.set = descriptor_pool.allocate(descriptor_set_layout);
	    VB_ASSERT(image.set);
	    VkDescriptorImageInfo image_info = {
		.sampler = sampler,
		.imageView = image.texture.image_view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	    };
	    VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = image.set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &image_info,
	    };
	    vkUpdateDescriptorSets(ctx->device, 1, &write, 0, nullptr);
	}
    }

    void GLTF::setup_sampler() {
	VkPhysicalDeviceProperties pdev_prop{};
	vkGetPhysicalDeviceProperties(ctx->physical_device, &pdev_prop);
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
    	VB_ASSERT(vkCreateSampler(ctx->device, &sampler_info, nullptr, &sampler) == VK_SUCCESS);
    }

    void GLTF::load_material(const fastgltf::Asset& asset, fastgltf::Material& material) {
	Material n_material;
    	if(material.pbrData.baseColorTexture.has_value()) {
	    n_material.base_color_tex_index = material.pbrData.baseColorTexture->textureIndex;
	}
	materials.push_back(n_material);
    }

    void GLTF::load_image(const fastgltf::Asset& asset, fastgltf::Image& image,
	    const std::filesystem::path& parent_path, vb::CommandPool& pool) {
	vb::Image n_image {ctx};
	int w, h, c;
	std::visit(fastgltf::visitor{[](auto& arg) {},
	    [&](fastgltf::sources::URI& filepath) {
		auto path = std::format("{}/{}", parent_path.c_str(), filepath.uri.c_str());
		VB_ASSERT(filepath.fileByteOffset == 0);
		VB_ASSERT(filepath.uri.isLocalPath());
		auto data = stbi_load(path.c_str(), &w, &h, &c, 4);
		if(!data) {
		    vb::log(std::format("failed to load image with stbi from path: {}",
			path.c_str()));
		    return;
		}
		VkExtent3D size = {(uint32_t)w, (uint32_t)h, 1};
		n_image.create(pool, data, size);
		VB_ASSERT(n_image.all_valid());
		stbi_image_free(data);
		images.push_back({n_image});
	    },
	    [&](fastgltf::sources::Vector& v) {
		auto data = stbi_load_from_memory((stbi_uc*)v.bytes.data(), (int)v.bytes.size(),
		    &w, &h, &c, 4);
		if(!data) {
		    vb::log("failed to load image from memory");
		    return;
		}
		VkExtent3D size = {(uint32_t)w, (uint32_t)h, 1};
		n_image.create(pool, data, size);
		VB_ASSERT(n_image.all_valid());
		stbi_image_free(data);
		images.push_back({n_image});
	    },
	    [&](fastgltf::sources::BufferView& view) {
		auto& bufferview = asset.bufferViews[view.bufferViewIndex];
		auto& buffer = asset.buffers[bufferview.bufferIndex];
		std::visit(fastgltf::visitor{[](auto& arg) {},
		    [&](fastgltf::sources::Vector& v) {
			auto data = stbi_load_from_memory(
		    	    (stbi_uc*)v.bytes.data() + bufferview.byteOffset,
			    (int)bufferview.byteLength, &w, &h, &c, 4);
			if(!data) {
			    vb::log("failed to load image from memory");
			    return;
			}
			VkExtent3D size = {(uint32_t)w, (uint32_t)h, 1};
			n_image.create(pool, data, size);
			VB_ASSERT(n_image.all_valid());
			stbi_image_free(data);
			images.push_back({n_image});
		    }
	       	}, buffer.data);
	    },
	}, image.data);
    }

    void GLTF::load_node(const fastgltf::Node& node_in, GLTF::Node* parent, 
		const fastgltf::Asset& asset, std::vector<Vertex>& vertex_vec,
		std::vector<uint32_t>& index_vec) {
	auto node = new GLTF::Node{};
	node->matrix = glm::mat4(1.0f);
	node->parent = parent;
	vb::log(std::format("node: {}",node_in.name));

	auto trs = std::get<fastgltf::TRS>(node_in.transform);
	node->matrix = glm::translate(node->matrix,
	    glm::vec3(glm::make_vec3(trs.translation.data())));
	auto quat = glm::make_quat(trs.rotation.data());
	node->matrix *= glm::mat4(quat);
	node->matrix = glm::scale(node->matrix, glm::vec3(glm::make_vec3(trs.scale.data())));

	if(node_in.children.size() > 0) {
	    for(const auto& child_idx: node_in.children)
		load_node(asset.nodes[child_idx], node, asset, vertex_vec, index_vec);
	}

	if(node_in.meshIndex.has_value()) {
	    const auto& mesh = asset.meshes[node_in.meshIndex.value()];
	    for(const auto& prim: mesh.primitives) {
		uint32_t first_index = index_vec.size();
		uint32_t vertex_start = vertex_vec.size();
		uint32_t index_count = asset.accessors[prim.indicesAccessor.value()].count;
		size_t v_idx = 0;
	    	{ // INDEX
	    	   auto& acc = asset.accessors[prim.indicesAccessor.value()];
	    	   index_vec.reserve(index_vec.size() + acc.count);
	    	   fastgltf::iterateAccessor<uint32_t>(asset, acc, [&](uint32_t i) {
	    	       index_vec.push_back(i + vertex_start);
	    	   });
	    	} { // POSITION
	    	   auto& acc = asset.accessors[prim.findAttribute("POSITION")->accessorIndex];
	    	   vertex_vec.resize(vertex_vec.size() + acc.count);
	    	   fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, acc,
	    	   	[&](glm::vec3 v, size_t i) {
			v_idx = i;
	    	       Vertex vert = {
	    	   	.position = v,
	    	   	.normal = {1,0,0},
	    	   	.color = glm::vec4{1.0f},
	    	       };
	    	       vertex_vec[vertex_start + i] = vert;
	    	   });
	    	} { // NORMALS
	    	   auto att = prim.findAttribute("NORMAL");
	    	   if(att != prim.attributes.end()) {
	    	       fastgltf::iterateAccessorWithIndex<glm::vec3>(asset,
	    	   	asset.accessors[att->accessorIndex], [&](glm::vec3 v, size_t i) {
	    	   	vertex_vec[vertex_start + i].normal = v;
			vertex_vec[vertex_start + i].color = glm::vec4(v.x, v.y, v.z, 1.0f);
	    	       });
	    	   }
	    	} { // UV
	    	   auto att = prim.findAttribute("TEXCOORD_0");
	    	   if(att != prim.attributes.end()) {
	    	       fastgltf::iterateAccessorWithIndex<glm::vec2>(asset,
	    	   	asset.accessors[att->accessorIndex], [&](glm::vec2 v, size_t i) {
	    	   	vertex_vec[vertex_start + i].uv_x = v.x;
	    	   	vertex_vec[vertex_start + i].uv_y = v.y;
	    	       });
	    	   }
	    	} { // COLOR
	    	   auto att = prim.findAttribute("COLOR_0");
	    	   if(att != prim.attributes.end()) {
		       if(asset.accessors[att->accessorIndex].type == fastgltf::AccessorType::Vec4) {
			    fastgltf::iterateAccessorWithIndex<glm::vec4>(asset,
				asset.accessors[att->accessorIndex], [&](glm::vec4 v, size_t i) {
				vertex_vec[vertex_start + i].color = v;
			    });
		       } else if(asset.accessors[att->accessorIndex].type
			       == fastgltf::AccessorType::Vec3) {
			    fastgltf::iterateAccessorWithIndex<glm::vec3>(asset,
				asset.accessors[att->accessorIndex], [&](glm::vec3 v, size_t i) {
				vertex_vec[vertex_start + i].color = glm::vec4(v.x, v.y, v.z, 1.0f);
			    });
		       }
	    	   }
	    	}
		Primitive primitive = {
		    .first_index = first_index,
		    .index_count = index_count,
		};
		if(prim.materialIndex.has_value())
		    primitive.material_index = prim.materialIndex.value();
		node->mesh.primitives.push_back(primitive);
	    }
	}
	if(parent) parent->children.push_back(node);
	else nodes.push_back(node);
    }

    void App::init_imgui() {
	vb::DescriptorAllocator builder {&vbc};
	vb::DescriptorAllocator::Ratio pool_sizes[] = {
    	    {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
    	    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
    	    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
    	    {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
    	    {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
    	    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
    	    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
    	    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
    	    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
    	    {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000},
    	};
	imgui_descriptor_pool = builder.create_pool(pool_sizes, 1000,
		VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
	ImGui::CreateContext();
    	VB_ASSERT(ImGui_ImplSDL3_InitForVulkan(vbc.window));
    	ImGui_ImplVulkan_InitInfo init_info = {
    	    .Instance = vbc.instance,
    	    .PhysicalDevice = vbc.physical_device,
    	    .Device = vbc.device,
    	    .Queue = vbc.queues_info.graphics_queue,
    	    .DescriptorPool = imgui_descriptor_pool,
    	    .MinImageCount = (uint32_t)vbc.swapchain_images.size(),
    	    .ImageCount = (uint32_t)vbc.swapchain_images.size(),
    	    .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
    	    .UseDynamicRendering = true,
    	    .PipelineRenderingCreateInfo = {
    	        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
    	        .colorAttachmentCount = 1,
    	        .pColorAttachmentFormats = &vbc.swapchain_format,
    	    },
    	};
    	VB_ASSERT(ImGui_ImplVulkan_Init(&init_info));
    	VB_ASSERT(ImGui_ImplVulkan_CreateFontsTexture());
    }

    VkImageLayout App::render_imgui(VkCommandBuffer cmd, 
	    VkImageLayout input_layout, uint32_t index) {
	vb::transition_image(cmd, vbc.swapchain_images[index],
		input_layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	VkRenderingAttachmentInfo color_attach = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
	    .imageView = vbc.swapchain_image_views[index],
	    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	    .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	};
	VkRenderingInfo rendering = {
	    .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
	    .renderArea = {{0,0}, vbc.swapchain_extent},
	    .layerCount = 1,
	    .colorAttachmentCount = 1,
	    .pColorAttachments = &color_attach,
	};
	vkCmdBeginRendering(cmd, &rendering);
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
	vkCmdEndRendering(cmd);
	return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    void App::imgui_interface() {
	//ImGui::ShowDemoWindow();
	ImGui::SetNextWindowSize(ImVec2(300,600));
	ImGui::Begin("vkgfxrenderer");
	ImGui::Text("[Q] lock camera");
	// STATISTICS
	ImGui::SeparatorText("statistics");
	ImGui::Text("fps:        %ld", stats.fps);
	ImGui::Text("frame time: %.3f ms", stats.frametime);
	ImGui::Text("draw time:  %.3f ms", stats.draw_time);
	ImGui::Text("triangles:  %ld", stats.triangles);
	ImGui::Text("draw calls: %ld", stats.drawcalls);
	ImGui::Separator();
	// SCREENSHOT
	static std::string screenshot_filename = "";
	if(ImGui::Button("screenshot")) {
	    screenshot_filename = save_screenshot(render_target.image);
	    ImGui::OpenPopup("save_screenshot");
	}
	if(ImGui::BeginPopupContextItem("save_screenshot")) {
	    ImGui::Text("saved to %s", screenshot_filename.c_str());
	    if(ImGui::Button("close")) ImGui::CloseCurrentPopup();
	    ImGui::EndPopup();
	}
	ImGui::End();
    }

    std::string App::save_screenshot(VkImage source) {
	vb::log("Saving screenshot of render target...");
	VkFormatProperties format_p;
	VkExtent3D size = {render_extent.width, render_extent.height, 1};
	VkImageCreateInfo image_info = {
	    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
	    .imageType = VK_IMAGE_TYPE_2D,
	    .format = VK_FORMAT_R8G8B8A8_SRGB,
	    .extent = size,
	    .mipLevels = 1,
	    .arrayLayers = 1,
	    .samples = VK_SAMPLE_COUNT_1_BIT,
	    .tiling = VK_IMAGE_TILING_LINEAR,
	    .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	};
	VmaAllocationCreateInfo allocation_info = {
	    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		    | VMA_ALLOCATION_CREATE_MAPPED_BIT,
	    .usage = VMA_MEMORY_USAGE_AUTO,
	    .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	};
	VkImage image;
	VmaAllocation allocation;
	VmaAllocationInfo info;
	if(vmaCreateImage(vbc.vma_allocator, &image_info, &allocation_info,
		    &image, &allocation, &info) != VK_SUCCESS) {
	    vb::log("Failed to create VkImage for screenshot target");
	    return "";
	}
	cmdpool.submit_command_buffer_to_queue(global_cmd_buffer,
    	    [&](VkCommandBuffer cmd) {
		vb::transition_image(cmd, source,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		vb::transition_image(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		vb::blit_image(cmd, source, image, size, size);
		vb::transition_image(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_GENERAL);
	});
	VkImageSubresource subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
	VkSubresourceLayout subres_layout;
	vkGetImageSubresourceLayout(vbc.device, image, &subresource, &subres_layout);
	char* data = (char*)info.pMappedData;
	data += subres_layout.offset;
	auto now_tp = std::chrono::system_clock::now();
	auto filename = std::format("{:%d%m%Y%H%M%OS}.png", now_tp);
	VB_ASSERT(stbi_write_png(filename.c_str(), size.width, size.height, 4, data,
		    subres_layout.rowPitch) != 0);
	vmaDestroyImage(vbc.vma_allocator, image, allocation);
	vb::log(std::format("Saved screenshot: {}", filename));
	return filename;
    }

int main() {
    App app {};
    app.run();
}
