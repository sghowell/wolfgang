#pragma once

#include <stdexcept>
#include <string>

namespace wolfgang {

// Shared native exception base reserved for binding-boundary validation errors.
// The current bindings translate standard invalid-input failures to ValueError;
// later phases can reuse this base for domain-specific exception plumbing.
class FastPauliError : public std::runtime_error {
public:
  explicit FastPauliError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace wolfgang


