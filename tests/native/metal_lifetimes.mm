#include "metal/device_pauli_sum_metal.hpp"

#include <stdexcept>

static int destroyed = 0;
@interface LifetimeProbe : NSObject
@end
@implementation LifetimeProbe
- (void)dealloc {
  ++destroyed;
  [super dealloc];
}
@end

int main() {
  // Unlike a non-ARC @autoreleasepool block, RAII drains during C++ unwinding.
  for (int iteration = 0; iteration < 50; ++iteration) {
    try {
      wolfgang::metal_detail::ScopedAutoreleasePool pool;
      [[[LifetimeProbe alloc] init] autorelease];
      throw std::runtime_error("simulated allocation or validation failure");
    } catch (const std::runtime_error&) {
    }
    if (destroyed != iteration + 1) return 1;
  }
}
