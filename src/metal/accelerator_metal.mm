#include "device_pauli_sum_metal.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace wolfgang::metal_detail {

std::string nsstring_to_string(NSString* value) {
  if (value == nil) {
    return {};
  }
  const char* utf8 = [value UTF8String];
  return utf8 == nullptr ? std::string{} : std::string(utf8);
}

id<MTLDevice> create_default_device() {
  return MTLCreateSystemDefaultDevice();
}

id<MTLDevice> create_default_device_or_throw() {
  id<MTLDevice> device = create_default_device();
  if (device == nil) {
    throw std::runtime_error("no Metal device is available");
  }
  return device;
}

void validate_device_ordinal(int device) {
  if (device != 0) {
    throw std::invalid_argument("Metal device ordinal is out of range");
  }
  id<MTLDevice> default_device = create_default_device();
  if (default_device == nil) {
    throw std::runtime_error("no Metal device is available");
  }
  [default_device release];
}

id<MTLCommandQueue> make_command_queue(id<MTLDevice> device) {
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (queue == nil) {
    throw std::runtime_error("Metal failed to create a command queue");
  }
  return queue;
}

id<MTLBuffer> make_shared_buffer(
    id<MTLDevice> device,
    const void* source,
    std::size_t bytes,
    const char* name) {
  if (bytes == 0) {
    return nil;
  }
  id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error(std::string("Metal failed to allocate shared buffer for ") + name);
  }
  if (source != nullptr) {
    std::memcpy([buffer contents], source, bytes);
    [buffer didModifyRange:NSMakeRange(0, bytes)];
  }
  return buffer;
}

id<MTLBuffer> make_private_buffer(
    id<MTLDevice> device,
    std::size_t bytes,
    const char* name) {
  if (bytes == 0) {
    return nil;
  }
  id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModePrivate];
  if (buffer == nil) {
    throw std::runtime_error(std::string("Metal failed to allocate private buffer for ") + name);
  }
  return buffer;
}

id<MTLComputePipelineState> make_compute_pipeline(
    id<MTLDevice> device,
    NSString* kernel_name,
    NSString* source) {
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
  if (library == nil) {
    throw std::runtime_error(
        "Metal failed to compile kernel library: " + nsstring_to_string([error localizedDescription]));
  }

  id<MTLFunction> function = [library newFunctionWithName:kernel_name];
  if (function == nil) {
    [library release];
    throw std::runtime_error("Metal kernel function is missing: " + nsstring_to_string(kernel_name));
  }

  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  [library release];
  if (pipeline == nil) {
    throw std::runtime_error(
        "Metal failed to create compute pipeline: " + nsstring_to_string([error localizedDescription]));
  }
  return pipeline;
}

id<MTLComputePipelineState> make_compute_pipeline_from_metallib(
    id<MTLDevice> device,
    NSString* kernel_name,
    const std::string& library_path) {
  NSError* error = nil;
  NSString* path = [NSString stringWithUTF8String:library_path.c_str()];
  if (path == nil || [path length] == 0) {
    throw std::invalid_argument("Metal offline library path is empty");
  }
  id<MTLLibrary> library = [device newLibraryWithFile:path error:&error];
  if (library == nil) {
    throw std::runtime_error(
        "Metal failed to load offline kernel library: " +
        nsstring_to_string([error localizedDescription]));
  }

  id<MTLFunction> function = [library newFunctionWithName:kernel_name];
  if (function == nil) {
    [library release];
    throw std::runtime_error(
        "Metal offline kernel function is missing: " + nsstring_to_string(kernel_name));
  }

  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  [library release];
  if (pipeline == nil) {
    throw std::runtime_error(
        "Metal failed to create offline compute pipeline: " +
        nsstring_to_string([error localizedDescription]));
  }
  return pipeline;
}

void wait_for_completion(id<MTLCommandBuffer> command_buffer, const char* action) {
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if ([command_buffer status] == MTLCommandBufferStatusCompleted) {
    return;
  }
  NSError* error = [command_buffer error];
  throw std::runtime_error(
      std::string("Metal ") + action + " failed: " +
      (error == nil ? std::string("unknown command buffer error")
                    : nsstring_to_string([error localizedDescription])));
}

[[noreturn]] void throw_unsupported_operation(const char* operation) {
  throw std::runtime_error(
      std::string("Metal backend does not implement ") + operation +
      " yet; use the CPU, CUDA, or HIP backend for this operation.");
}

}  // namespace wolfgang::metal_detail
