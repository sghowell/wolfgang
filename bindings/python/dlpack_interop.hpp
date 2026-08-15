#pragma once

#include "dlpack/dlpack.h"

#include <optional>

namespace wolfgang::python::detail {

constexpr DLPackVersion kSupportedDLPackVersion{
    DLPACK_MAJOR_VERSION,
    DLPACK_MINOR_VERSION,
};
constexpr DLPackVersion kMinimumReadOnlyDLPackVersion{1, 0};

[[nodiscard]] constexpr bool dlpack_version_less(
    DLPackVersion lhs,
    DLPackVersion rhs) noexcept {
  return lhs.major < rhs.major || (lhs.major == rhs.major && lhs.minor < rhs.minor);
}

// Versioned tensors and their flags were introduced in DLPack 1.0. Returning
// no version therefore means this producer cannot preserve its read-only
// contract for the consumer's requested maximum.
[[nodiscard]] constexpr std::optional<DLPackVersion> negotiate_dlpack_version(
    DLPackVersion consumer_max) noexcept {
  if (dlpack_version_less(consumer_max, kMinimumReadOnlyDLPackVersion)) {
    return std::nullopt;
  }
  return dlpack_version_less(consumer_max, kSupportedDLPackVersion)
      ? consumer_max
      : kSupportedDLPackVersion;
}

}  // namespace wolfgang::python::detail
