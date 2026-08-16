#include "bindings.hpp"

#include <nanobind/nanobind.h>

NB_MODULE(_wolfgang_core, module) {
  module.doc() = "Native extension for Wolfgang packed Pauli-sum storage and I/O.";
  wolfgang::python::register_stable_bindings(module);
#if WOLFGANG_ENABLE_INTERNAL_BINDINGS
  wolfgang::python::register_internal_bindings(module);
#endif
}
