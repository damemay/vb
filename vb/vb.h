#pragma once

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <string>
#include <functional>
#include <deque>
#include <optional>
#include <span>
#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#ifndef NDEBUG
#define VB_ASSERT(FN) assert(FN);
#else
#define VB_ASSERT(FN) if(!(FN)) { vb::log(#FN " failed"); }
#endif

namespace vb {
    struct Context;
    struct ContextDependant {
	Context* ctx;
    };

    static inline void log(const std::string& buf) {
	fprintf(stderr, "vor: %s\n", buf.c_str());
    }

    struct DeletionQueue {
	std::deque<std::function<void()>> deletors;
	void push(std::function<void()>&& fn) { deletors.push_back(fn); }
	void flush() {
	    for(auto i = deletors.rbegin(); i != deletors.rend(); i++)
		(*i)();
	    deletors.clear();
	}
    };
}

namespace vb {
    struct QuickCommand {
	VkCommandPool cmd_pool;
	VkCommandBuffer cmd_buffer;
	VkFence fence;
    };

    struct Frame {
	VkCommandPool cmd_pool;
	VkCommandBuffer cmd_buffer;

	VkSemaphore image_available_semaphore;
	VkSemaphore finish_render_semaphore;
	VkFence render_fence;
    };

    struct QueuesInfo {
	VkQueue graphics_queue;
    	VkQueue compute_queue;
    	VkQueue present_queue;

    	uint32_t graphics_index;
    	uint32_t compute_index;
    	uint32_t present_index;
    };

    struct VulkanExtensionSupport {
	bool acceleration_structure {false};
	bool raytracing_pipeline {false};
	bool rayquery {false};
	bool pipeline_library {false};
	bool deferred_host_operations {false};
	bool meshshader {false};
    };

    struct Context {
	struct Info {
	    std::string title;
	    uint32_t width;
    	    uint32_t height;
	    SDL_InitFlags sdl3_init_flags;
	    SDL_WindowFlags sdl3_window_flags;
    	};

	Info info;
	SDL_Window* window {nullptr};

	VulkanExtensionSupport extension_support{};
	bool raytracing {false};
	bool meshshader {false};

	VmaAllocator vma_allocator;
	VkInstance instance;
	VkPhysicalDevice physical_device {VK_NULL_HANDLE};
	QueuesInfo queues_info;
	VkDevice device;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapchain;
	VkFormat swapchain_format;
	VkExtent2D swapchain_extent;
	std::vector<VkImage> swapchain_images;
	std::vector<VkImageView> swapchain_image_views;
	std::vector<VkFramebuffer> swapchain_framebuffers;

	float render_aspect_ratio {0.0f};

	bool resize {false};
	static constexpr uint8_t max_frames {2};
	Frame frames[max_frames];
	uint8_t frame_index {0};

	QuickCommand quick_command_info;

    	static constexpr const char* validation_layer_name[1] {"VK_LAYER_KHRONOS_validation"};
	bool validation_layers_support {false};
#ifndef NDEBUG
	VkDebugUtilsMessengerEXT debug_messenger;
#endif

	Context(const Info& context_info);
	~Context();

	void create_swapchain_framebuffers(VkRenderPass render_pass, std::vector<VkImageView> attachments = {});
	void destroy_swapchain_framebuffers();
	void recreate_swapchain(VkRenderPass render_pass);

	void submit_quick_command(std::function<void(VkCommandBuffer cmd)>&& fn);

	inline Frame* get_current_frame() { return &frames[frame_index % max_frames]; }
	inline std::optional<uint32_t> wait_on_image_reset_fence(Frame* frame) {
	    vkWaitForFences(device, 1, &frame->render_fence, VK_TRUE, UINT64_MAX);
	    uint32_t image_index;
	    VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, frame->image_available_semaphore, VK_NULL_HANDLE, &image_index);
	    if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		resize = true;
		return std::nullopt;
	    }
	    vkResetFences(device, 1, &frame->render_fence);
	    return image_index;
	}

	private:
	    void create_instance();
	    void pick_physical_device();
	    void create_surface();
	    void create_device();
	    void create_swapchain(uint32_t width, uint32_t height);
	    void create_swapchain_image_views();
	    void destroy_swapchain();
	    void create_frames();
	    void init_vma();
	    void init_quick_cmd();
#ifndef NDEBUG
	    VkDebugUtilsMessengerCreateInfoEXT fill_debug_messenger_create_info();
	    void create_debug_messenger();
	    bool test_for_validation_layers();
#endif
    };
}

namespace vb::render {
    inline void begin_reset_command_buffer(VkCommandBuffer cmd) {
	vkResetCommandBuffer(cmd, 0);
	VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	VB_ASSERT(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
    }

    inline void end_command_buffer(VkCommandBuffer cmd) {VB_ASSERT(vkEndCommandBuffer(cmd) == VK_SUCCESS);}

    inline void submit_queue(VkQueue queue, VkFence fence,
	    std::vector<VkCommandBuffer> buffers, std::vector<VkSemaphore> wait_semaphores,
	    std::vector<VkSemaphore> signal_semaphores, std::vector<VkPipelineStageFlags> wait_destination_stage) {
	VkPipelineStageFlags wait[1] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
	VkSubmitInfo submit = {
	    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
	    .waitSemaphoreCount = (uint32_t)wait_semaphores.size(),
	    .pWaitSemaphores = wait_semaphores.data(),
	    .pWaitDstStageMask = wait_destination_stage.data(),
	    .commandBufferCount = (uint32_t)buffers.size(),
	    .pCommandBuffers = buffers.data(),
	    .signalSemaphoreCount = (uint32_t)signal_semaphores.size(),
	    .pSignalSemaphores = signal_semaphores.data(),
	};
	VB_ASSERT(vkQueueSubmit(queue, 1, &submit, fence) == VK_SUCCESS);
    }

    inline void present_queue(VkQueue queue, VkSwapchainKHR* swapchain, std::vector<VkSemaphore> wait_semaphores, 
	    uint32_t* index) {
	VkPresentInfoKHR present = {
	    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
	    .waitSemaphoreCount = (uint32_t)wait_semaphores.size(),
	    .pWaitSemaphores = wait_semaphores.data(),
	    .swapchainCount = 1,
	    .pSwapchains = swapchain,
	    .pImageIndices = index,
	};
	vkQueuePresentKHR(queue, &present);
    }
}

namespace vb::fill {
    VkCommandBufferAllocateInfo cmd_buffer_allocate_info(VkCommandPool pool, uint32_t count = 1);

    VkRenderingAttachmentInfo attachment_info(VkImageView image_view, VkClearValue* clear,
	    VkImageLayout image_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depth_attachment_info(VkImageView image_view,
	    VkImageLayout image_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    VkRenderingInfo rendering_info(VkExtent2D render_extent,
	    VkRenderingAttachmentInfo* color_attachment,
	    VkRenderingAttachmentInfo* depth_attachment);
    VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspect_mask);
}

namespace vb::create {
    VkCommandPool cmd_pool(VkDevice device, uint32_t queue_family_index,
	    VkCommandPoolCreateFlags flags);
    VkSemaphore semaphore(VkDevice device, VkSemaphoreCreateFlags flags = 0);
    VkFence fence(VkDevice device, VkFenceCreateFlags flags = 0);
    VkDescriptorSetLayout descriptor_set_layout(VkDevice device,
	    std::vector<VkDescriptorSetLayoutBinding> bindings,
	    VkShaderStageFlags stages,
	    VkDescriptorSetLayoutCreateFlags flags,
	    void* next);
    VkPipeline compute_pipeline(VkDevice device,
	    VkPipelineLayout layout,
	    VkShaderModule shader_module);

    std::optional<VkShaderModule> shader_module(VkDevice device, const char* path);

    struct OptionalValidator {virtual bool all_valid() = 0;};

    struct Descriptor : public ContextDependant {
	struct Ratio {
	    VkDescriptorType type;
	    float ratio;
	};

	Descriptor(Context* context): ContextDependant{context} {}
	void create(std::span<Ratio> pool_ratios, uint32_t init_sets = 1000, VkDescriptorPoolCreateFlags flags = 0);
	std::optional<VkDescriptorSet> allocate(VkDescriptorSetLayout layout, void* next = nullptr);
	void flush();
	void clean();

	private:
    	    uint32_t sets;
    	    std::vector<Ratio> ratios;
    	    std::vector<VkDescriptorPool> full_pools;
    	    std::vector<VkDescriptorPool> ready_pools;

	    std::optional<VkDescriptorPool> get_pool();
	    std::optional<VkDescriptorPool> create_pool(std::span<Ratio> pool_ratios, uint32_t init_sets, VkDescriptorPoolCreateFlags flags = 0);
    };

    struct Buffer: public ContextDependant, public OptionalValidator {
	std::optional<VkBuffer> buffer {std::nullopt};
	std::optional<VmaAllocation> allocation {std::nullopt};
	std::optional<VmaAllocationInfo> info {std::nullopt};
	bool all_valid() {return buffer.has_value() && allocation.has_value() && info.has_value();}

	Buffer(Context* context): ContextDependant{context} {}
	void create(const size_t size, VkBufferCreateFlags usage, VmaMemoryUsage mem_usage);
	void clean();
    };

    struct Image: public ContextDependant, public OptionalValidator {
	std::optional<VkImage> image {std::nullopt};
	std::optional<VkImageView> image_view {std::nullopt};
	std::optional<VmaAllocation> allocation {std::nullopt};
	bool all_valid() {return image.has_value() && image_view.has_value() && allocation.has_value();}

	VkExtent3D extent;
	VkFormat format;
	Image(Context* context): ContextDependant{context} {}
	void create(VkExtent3D extent, VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT,
		VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT 
		| VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		bool mipmap = false);
	void create(void* data, VkExtent3D extent, VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT,
		VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT 
		| VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		bool mipmap = false);
	void clean();
    };

//     struct Subpass : public OptionalValidator {
// 	std::optional<VkSubpassDescription> description {std::nullopt};
// 	bool all_valid() {return description.has_value();}
// 
// 	struct Attachment {
// 	    VkAttachmentDescription description;
// 	    VkAttachmentReference reference;
// 	};
// 	uint32_t attachment_index {0};
// 	std::vector<Attachment> input_attachments;
// 	std::vector<Attachment> color_attachments;
// 	std::vector<Attachment> resolve_attachments;
// 	std::vector<Attachment> depth_attachment;
// 	std::vector<uint32_t> preserve_attachments;
// 
// #define VB_ADD_ATTACHMENT(NAME, VECTOR) \
//     void NAME(VkAttachmentDescription& description, VkImageLayout subpass_layout) { \
// 	VkAttachmentReference ref { attachment_index++, subpass_layout }; \
// 	VECTOR.push_back({description, ref}); \
//     }
// 	VB_ADD_ATTACHMENT(add_input_attachment, input_attachments)
//     	VB_ADD_ATTACHMENT(add_color_attachment, color_attachments)
//     	VB_ADD_ATTACHMENT(add_resolve_attachment, resolve_attachments)
//     	VB_ADD_ATTACHMENT(add_depth_attachment, depth_attachment)
// #undef VB_ADD_ATTACHMENT
// 	void add_preserve_attachment(const uint32_t index) {preserve_attachments.push_back(index);}
// #define VB_GET_REFERENCES(NAME, VECTOR) \
//     std::vector<VkAttachmentReference> NAME() { \
// 	std::vector<VkAttachmentReference> references; \
// 	if(VECTOR.size() != 0) for(auto attachment: VECTOR) \
//     	    references.push_back(attachment.reference); \
// 	return references; \
//     }
//     VB_GET_REFERENCES(get_input_references, input_attachments)
//     VB_GET_REFERENCES(get_color_references, color_attachments)
//     VB_GET_REFERENCES(get_resolve_references, resolve_attachments)
//     VB_GET_REFERENCES(get_depth_references, depth_attachment)
// #undef VB_GET_REFERENCE
// 	void create(VkPipelineBindPoint bind_point, VkSubpassDescriptionFlags flags = 0);
//     };
// 
//     struct RenderPass: public ContextDependant, public OptionalValidator {
// 	std::optional<VkRenderPass> render_pass {std::nullopt};
// 	bool all_valid() {return render_pass.has_value();}
// 
// 	std::vector<VkSubpassDependency> subpass_dependencies;
// 
// 	void add_subpassk
//     };

    struct GraphicsPipeline: public ContextDependant, public OptionalValidator {
	private:
	    std::vector<VkShaderModule> shader_modules;
	    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
	    std::vector<VkPushConstantRange> push_constants;

	    VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	    };
	    VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
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
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f,
	    };

	public:
	    GraphicsPipeline(Context* context): ContextDependant{context} {}

	    void add_shader(VkShaderModule& shader_module, VkShaderStageFlagBits stage);
	    void add_shader(const char* path, VkShaderStageFlagBits stage);

	    void add_push_constant(const uint32_t size, VkShaderStageFlagBits stage, const uint32_t offset = 0);

	    // input assembly
	    inline void set_topology(VkPrimitiveTopology topology) { input_assembly.topology = topology; }
	    // rasterization
	    inline void set_polygon_mode(VkPolygonMode mode) { rasterization.polygonMode = mode; }
	    inline void set_cull_mode(VkCullModeFlags mode) { rasterization.cullMode = mode; }
	    inline void set_front_face(VkFrontFace face) { rasterization.frontFace = face; }
	    // multisample
	    inline void set_sample_count(VkSampleCountFlagBits count) { multisample.rasterizationSamples = count; }
	    inline void enable_sample_shading(float min_sample = 1.0f) { multisample.sampleShadingEnable = VK_TRUE; multisample.minSampleShading = min_sample; }
	    // detph stencil
	    inline void enable_depth_test() { depth_stencil.depthTestEnable = VK_TRUE; depth_stencil.depthWriteEnable = VK_TRUE; }
	    inline void set_depth_comparison(VkCompareOp operation) { depth_stencil.depthCompareOp =operation; }
	    inline void enable_depth_bounds_test() { depth_stencil.depthBoundsTestEnable = VK_TRUE; }
	    inline void set_depth_bounds(float min, float max) { depth_stencil.minDepthBounds = min; depth_stencil.maxDepthBounds = max; }
	    inline void enable_stencil_test() { depth_stencil.stencilTestEnable = VK_TRUE; }
	    inline void set_stencil_operations(VkStencilOpState front, VkStencilOpState back) { depth_stencil.front = front; depth_stencil.back = back; }

	    std::optional<VkPipelineLayout> layout {std::nullopt};
	    std::optional<VkPipeline> pipeline {std::nullopt};
	    bool all_valid() {return layout.has_value() && pipeline.has_value();}

	    void create(VkRenderPass render_pass, uint32_t subpass_index);
	    void clean();
    };
}

