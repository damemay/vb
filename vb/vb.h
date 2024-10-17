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
#define VB_ASSERT(FN) if(!(FN)) { vb::log(#FN " failed"); exit(1); }
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

    struct Context {
	constexpr static uint32_t api_version = VK_API_VERSION_1_3;

	struct Info {
	    std::string title;
	    uint32_t width;
    	    uint32_t height;
	    SDL_InitFlags sdl3_init_flags;
	    SDL_WindowFlags sdl3_window_flags;

	    std::vector<const char*> required_extensions;
	    std::vector<const char*> optional_extensions;
	    bool enable_all_available_extensions = false;

	    VkPhysicalDeviceFeatures vk10features;
	    VkPhysicalDeviceVulkan11Features vk11features;
	    VkPhysicalDeviceVulkan12Features vk12features;
	    VkPhysicalDeviceVulkan13Features vk13features;
	    bool enable_all_available_features = false;

	    VkSurfaceFormatKHR surface_format = {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
	    VkPresentModeKHR present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
    	};

	Info info;
	SDL_Window* window {nullptr};

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
	struct SwapchainSupportData {
	    VkSurfaceFormatKHR format;
	    VkPresentModeKHR present_mode;
	    VkSurfaceCapabilitiesKHR surface_capabilities;
	    uint32_t image_count;
	    VkSharingMode image_sharing_mode;
	    std::vector<uint32_t> queue_family_indices;
	};
	SwapchainSupportData swapchain_support_data;

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

	[[nodiscard]] Context(const Info& context_info);
	~Context();

	[[nodiscard]] const std::vector<const char*>& get_enabled_extensions() const { return requested_extensions; }

	void recreate_swapchain(std::function<void(uint32_t,uint32_t)>&& call_before_swapchain_create = nullptr);

	void submit_quick_command(std::function<void(VkCommandBuffer cmd)>&& fn);

	[[nodiscard]] inline Frame* get_current_frame() { return &frames[frame_index % max_frames]; }
	[[nodiscard]] inline std::optional<uint32_t> wait_on_image_reset_fence(Frame* frame) {
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
	    std::vector<const char*> available_extensions;
	    std::vector<const char*> requested_extensions;

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
	    [[nodiscard]] VkDebugUtilsMessengerCreateInfoEXT fill_debug_messenger_create_info();
	    void create_debug_messenger();
	    [[nodiscard]] bool test_for_validation_layers();
#endif
    };
}

namespace vb::sync {
    void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout);
    void transition_image2(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout);
}

namespace vb::fill {
    [[nodiscard]] VkCommandBufferAllocateInfo cmd_buffer_allocate_info(VkCommandPool pool, uint32_t count = 1);
    [[nodiscard]] VkRenderingAttachmentInfo attachment_info(VkImageView image_view, VkClearValue* clear,
	    VkImageLayout image_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    [[nodiscard]] VkRenderingAttachmentInfo depth_attachment_info(VkImageView image_view,
	    VkImageLayout image_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    [[nodiscard]] VkRenderingInfo rendering_info(VkExtent2D render_extent,
	    VkRenderingAttachmentInfo* color_attachment,
	    VkRenderingAttachmentInfo* depth_attachment);
    [[nodiscard]] VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspect_mask);
}

namespace vb::create {
    [[nodiscard]] VkCommandPool cmd_pool(VkDevice device, uint32_t queue_family_index,
	    VkCommandPoolCreateFlags flags);
    [[nodiscard]] VkSemaphore semaphore(VkDevice device, VkSemaphoreCreateFlags flags = 0);
    [[nodiscard]] VkFence fence(VkDevice device, VkFenceCreateFlags flags = 0);
    [[nodiscard]] VkDescriptorSetLayout descriptor_set_layout(VkDevice device,
	    std::vector<VkDescriptorSetLayoutBinding> bindings,
	    VkShaderStageFlags stages,
	    VkDescriptorSetLayoutCreateFlags flags,
	    void* next);
    [[nodiscard]] VkPipeline compute_pipeline(VkDevice device,
	    VkPipelineLayout layout,
	    VkShaderModule shader_module);
    [[nodiscard]] std::optional<VkShaderModule> shader_module(VkDevice device, const char* path);
}

namespace vb::builder {
    struct OptionalValidator {virtual bool all_valid() = 0;};
    struct Descriptor : public ContextDependant {
	struct Ratio {
	    VkDescriptorType type;
	    float ratio;
	};

	[[nodiscard]] Descriptor(Context* context): ContextDependant{context} {}
	void create(std::span<Ratio> pool_ratios, uint32_t init_sets = 1000, VkDescriptorPoolCreateFlags flags = 0);
	[[nodiscard]] std::optional<VkDescriptorSet> allocate(VkDescriptorSetLayout layout, void* next = nullptr);
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

	[[nodiscard]] Buffer(Context* context): ContextDependant{context} {}
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
	[[nodiscard]] Image(Context* context): ContextDependant{context} {}
	void create(VkExtent3D extent, VkFormat format = VK_FORMAT_R8G8B8A8_SRGB,
		VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT  | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		bool mipmap = false);
	void create(void* data, VkExtent3D extent, VkFormat format = VK_FORMAT_R8G8B8A8_SRGB,
		VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		bool mipmap = false);
	void clean();
    };

    struct GraphicsPipeline: public ContextDependant, public OptionalValidator {
	private:
	    std::vector<VkShaderModule> shader_modules;
	    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
	    std::vector<VkPushConstantRange> push_constants;
	    std::vector<VkDescriptorSetLayout> descriptor_set_layouts;

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
	    [[nodiscard]] GraphicsPipeline(Context* context): ContextDependant{context} {}

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

	    void create(VkRenderPass render_pass, uint32_t subpass_index, std::vector<VkDescriptorSetLayout> descriptor_layouts = {});
	    void clean();
    };
}

