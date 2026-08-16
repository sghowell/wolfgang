#include "bindings.hpp"

namespace wolfgang::python {

void bind_pauli_sum(nanobind::module_& module);
void bind_build_info(nanobind::module_& module);

void register_stable_bindings(nanobind::module_& module) {
  bind_pauli_sum(module);
  bind_build_info(module);
}

}  // namespace wolfgang::python
