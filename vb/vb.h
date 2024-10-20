#pragma once

#define VK_NO_PROTOTYPES
#include <string>
#include <functional>
#include <optional>
#include <span>
#include <assert.h>
#include <SDL3/SDL.h>
#include <vk_mem_alloc.h>
#include <volk.h>

#ifndef NDEBUG
#define VB_ASSERT(FN) assert(FN);
#else
#define VB_ASSERT(FN) if(!(FN)) { vb::log(#FN " failed"); exit(1); }
#endif

namespace vb {
    struct Context;
    struct ContextDependant {Context* ctx;};
    static inline void log(const std::string& buf) {fprintf(stderr, "vb: %s\n", buf.c_str());}

    struct Context {
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
	    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    	};
	uint32_t available_implementation_api_version {0};
    	uint32_t api_minor_version = 0;

	Info info;
	SDL_Window* window {nullptr};

	VmaAllocator vma_allocator;
	VkInstance instance;
	VkPhysicalDevice physical_device {VK_NULL_HANDLE};
	VkDevice device;
	VkSurfaceKHR surface;

	struct QueuesInfo {
    	    VkQueue graphics_queue;
    		VkQueue compute_queue;
    		VkQueue present_queue;

    		uint32_t graphics_index;
    		uint32_t compute_index;
    		uint32_t present_index;
    	};
	QueuesInfo queues_info;

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

    	static constexpr const char* validation_layer_name[1] {"VK_LAYER_KHRONOS_validation"};
	bool validation_layers_support {false};
#ifndef NDEBUG
	VkDebugUtilsMessengerEXT debug_messenger;
#endif

	std::function<void()> resize_callback {nullptr};

	[[nodiscard]] Context(const Info& context_info);
	~Context();

	void set_resize_callback(std::function<void()>&& fn) { resize_callback = fn; }

	[[nodiscard]] const std::vector<std::string>& get_enabled_extensions() const { return requested_extensions; }
	[[nodiscard]] const std::vector<std::string>& get_available_extensions() const { return available_extensions; }

	[[nodiscard]] std::optional<uint32_t> acquire_next_image(VkSemaphore signal_semaphore);
	void recreate_swapchain(std::function<void(uint32_t,uint32_t)>&& call_before_swapchain_create = nullptr);

	protected:
	    std::vector<std::string> available_extensions;
	    std::vector<std::string> requested_extensions;

	    void create_instance();
	    void pick_physical_device();
	    void create_surface();
	    void create_device();
	    void create_swapchain(uint32_t width, uint32_t height);
	    void create_swapchain_image_views();
	    void destroy_swapchain();
	    void init_vma();
#ifndef NDEBUG
	    [[nodiscard]] VkDebugUtilsMessengerCreateInfoEXT fill_debug_messenger_create_info();
	    void create_debug_messenger();
	    [[nodiscard]] bool test_for_validation_layers();
#endif
    };
}

namespace vb::sync {
    void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout);
    void blit_image(VkCommandBuffer cmd, VkImage source, VkImage dest, VkExtent3D src_extent,
	    VkExtent3D dst_extent, VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT);
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
    [[nodiscard]] VkShaderModule shader_module(VkDevice device, const char* path);
}

namespace vb::builder {
    struct OptionalValidator {virtual bool all_valid() = 0;};

    struct CommandPool : public ContextDependant, public OptionalValidator {
	VkCommandPool pool {VK_NULL_HANDLE};
	VkFence fence {VK_NULL_HANDLE};
	VkQueue queue {VK_NULL_HANDLE};
	uint32_t queue_index {0};
	bool all_valid() {return pool&&fence&&queue;}

	[[nodiscard]] CommandPool(Context* context): ContextDependant{context} {}
	void create(VkQueue queue, uint32_t queue_index,
		VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	[[nodiscard]] VkCommandBuffer allocate();
	void submit_command_buffer_to_queue(VkCommandBuffer cmd_buffer, std::function<void(VkCommandBuffer cmd)>&& fn);
	void clean();
    };

    struct Descriptor : public ContextDependant {
	struct Ratio {
	    VkDescriptorType type;
	    float ratio;
	};

	[[nodiscard]] Descriptor(Context* context): ContextDependant{context} {}
	void create(std::span<Ratio> pool_ratios, uint32_t init_sets = 1000, VkDescriptorPoolCreateFlags flags = 0);
	[[nodiscard]] VkDescriptorSet allocate(VkDescriptorSetLayout layout, void* next = nullptr);
	void flush();
	void clean();

    	uint32_t sets;
	std::vector<Ratio> ratios;
	std::vector<VkDescriptorPool> full_pools;
	std::vector<VkDescriptorPool> ready_pools;

	VkDescriptorPool get_pool();
	VkDescriptorPool create_pool(std::span<Ratio> pool_ratios, uint32_t init_sets, VkDescriptorPoolCreateFlags flags = 0);
    };

    struct Buffer: public ContextDependant, public OptionalValidator {
	VkBuffer buffer {VK_NULL_HANDLE};
	VmaAllocation allocation {VK_NULL_HANDLE};
	VmaAllocationInfo info;
	bool all_valid() {return buffer&&allocation;}

	[[nodiscard]] Buffer(Context* context): ContextDependant{context} {}
	void create(const size_t size, VkBufferCreateFlags usage, VmaMemoryUsage mem_usage);
	void clean();
    };

    struct Image: public ContextDependant, public OptionalValidator {
	VkImage image {VK_NULL_HANDLE};
	VkImageView image_view {VK_NULL_HANDLE};
	VmaAllocation allocation {VK_NULL_HANDLE};
	bool all_valid() {return image&&image_view&&allocation;}

	VkExtent3D extent;
	VkFormat format;
	[[nodiscard]] Image(Context* context): ContextDependant{context} {}
	void create(VkExtent3D extent, VkFormat format = VK_FORMAT_B8G8R8A8_SRGB,
		VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT  | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		bool mipmap = false);
	void create(CommandPool pool, void* data, VkExtent3D extent, VkFormat format = VK_FORMAT_R8G8B8A8_SRGB,
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

	    VkPipelineLayout layout {VK_NULL_HANDLE};
	    VkPipeline pipeline {VK_NULL_HANDLE};
	    bool all_valid() {return layout&&pipeline;}

	    void create(void* pNext, VkPipelineCreateFlags flags, VkRenderPass render_pass, uint32_t subpass_index, std::vector<VkDescriptorSetLayout> descriptor_layouts);
	    void create(VkRenderPass render_pass, uint32_t subpass_index = 0, std::vector<VkDescriptorSetLayout> descriptor_layouts = {});
	    void create(void* pNext, std::vector<VkDescriptorSetLayout> descriptor_layouts = {});
	    void create(void* pNext, VkPipelineCreateFlags flags = 0, std::vector<VkDescriptorSetLayout> descriptor_layouts = {});
	    void clean();
    };
}

