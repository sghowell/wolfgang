#pragma once

#include <nanobind/nanobind.h>

namespace wolfgang::python {

void register_stable_bindings(nanobind::module_& module);
void register_internal_bindings(nanobind::module_& module);

}  // namespace wolfgang::python
