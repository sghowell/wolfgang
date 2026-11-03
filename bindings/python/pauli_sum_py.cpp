#include "wolfgang/cpu_backend.hpp"
#include "wolfgang/device_commutation_matrix.hpp"
#include "wolfgang/device_pauli_sum.hpp"
#include "wolfgang/pauli_sum.hpp"

#include "dlpack/dlpack.h"
#include "dlpack_interop.hpp"

#if WOLFGANG_BUILD_CUDA_ENABLED
#include "cuda/device_commutation_matrix.cuh"
#endif

#include <algorithm>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace wolfgang::python {

namespace {

struct DeviceCommutationDlpackContext {
  PyObject* owner = nullptr;
  std::unique_ptr<std::int64_t[]> shape;
  std::unique_ptr<std::int64_t[]> strides;

  ~DeviceCommutationDlpackContext() {
    if (owner != nullptr && Py_IsInitialized()) {
      PyGILState_STATE state = PyGILState_Ensure();
      Py_DECREF(owner);
      PyGILState_Release(state);
    }
  }
};

// Use only for pure C++ computation. Arguments and buffers must already be
// validated and pinned; Python result conversion happens after reacquiring GIL.
template <typename Function>
auto cpu_without_gil(Function&& function) {
  nb::gil_scoped_release release;
  return std::forward<Function>(function)();
}

class PythonBufferView {
public:
  explicit PythonBufferView(nb::handle object) {
    if (PyObject_GetBuffer(
            object.ptr(),
            &view_,
            PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) != 0) {
      PyErr_Clear();
      throw nb::type_error("psi must be a NumPy array or another buffer-protocol object");
    }
  }

  PythonBufferView(const PythonBufferView&) = delete;
  PythonBufferView& operator=(const PythonBufferView&) = delete;

  ~PythonBufferView() { PyBuffer_Release(&view_); }

  [[nodiscard]] const Py_buffer& get() const noexcept { return view_; }

private:
  Py_buffer view_{};
};

// View a C-contiguous array the caller keeps alive. The buffer export is
// released immediately; the memory stays valid while the array object lives.
template <typename T>
std::span<const T> contiguous_span(nb::handle array, std::size_t count) {
  if (count == 0) {
    return {};
  }
  PythonBufferView buffer(array);
  return {static_cast<const T*>(buffer.get().buf), count};
}

class WritablePythonBufferView {
public:
  explicit WritablePythonBufferView(nb::handle object) {
    if (PyObject_GetBuffer(
            object.ptr(),
            &view_,
            PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) != 0) {
      throw nb::python_error();
    }
  }

  WritablePythonBufferView(const WritablePythonBufferView&) = delete;
  WritablePythonBufferView& operator=(const WritablePythonBufferView&) = delete;

  ~WritablePythonBufferView() { PyBuffer_Release(&view_); }

  [[nodiscard]] const Py_buffer& get() const noexcept { return view_; }

private:
  Py_buffer view_{};
};

std::size_t checked_size_from_signed(long long value, const char* name) {
  if (value < 0) {
    throw nb::value_error((std::string(name) + " must be non-negative").c_str());
  }
  return static_cast<std::size_t>(value);
}

std::size_t checked_size_from_python_int(nb::handle value, const char* name) {
  if (!PyLong_Check(value.ptr())) {
    throw nb::value_error((std::string(name) + " must be a non-negative integer").c_str());
  }
  unsigned long long raw_value = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw nb::value_error((std::string(name) + " must be a non-negative integer that fits size_t").c_str());
  }
  if (raw_value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
    throw nb::value_error((std::string(name) + " must fit size_t").c_str());
  }
  return static_cast<std::size_t>(raw_value);
}

void throw_buffer_error(const char* message) {
  PyErr_SetString(PyExc_BufferError, message);
  throw nb::python_error();
}

std::vector<std::string> parse_labels(nb::iterable labels_obj) {
  if (PyUnicode_Check(labels_obj.ptr())) {
    throw nb::value_error("labels must be an iterable of dense label strings, not a single string");
  }

  std::vector<std::string> labels;
  try {
    for (nb::handle item : labels_obj) {
      labels.push_back(nb::cast<std::string>(item));
    }
  } catch (const nb::cast_error&) {
    throw nb::value_error("labels must be an iterable of strings");
  }
  return labels;
}

std::complex<double> parse_complex_value(nb::handle value, const char* name) {
  try {
    return nb::cast<std::complex<double>>(value);
  } catch (const nb::cast_error&) {
    throw nb::value_error((std::string(name) + " must be numeric").c_str());
  }
}

std::vector<std::complex<double>> parse_coefficients(
    nb::object coeffs_obj,
    std::size_t expected_size) {
  if (coeffs_obj.is_none()) {
    return std::vector<std::complex<double>>(expected_size, std::complex<double>{1.0, 0.0});
  }

  try {
    std::complex<double> scalar = nb::cast<std::complex<double>>(coeffs_obj);
    if (expected_size != 1) {
      throw nb::value_error("scalar coeffs are valid only for a single label");
    }
    return {scalar};
  } catch (const nb::cast_error&) {
    // Non-scalar coefficient inputs are parsed as iterables below.
  }

  // One-dimensional numeric NumPy arrays convert in bulk; boxing one Python
  // complex per element dominated label construction for large operators.
  // Only real NumPy arrays take this path; other array-likes (CuPy, torch) keep
  // the element-wise path they used before.
  nb::module_ numpy = nb::module_::import_("numpy");
  if (PyObject_IsInstance(coeffs_obj.ptr(), numpy.attr("ndarray").ptr()) == 1 &&
      nb::cast<int>(coeffs_obj.attr("ndim")) == 1) {
    const std::string kind = nb::cast<std::string>(coeffs_obj.attr("dtype").attr("kind"));
    if (kind == "b" || kind == "i" || kind == "u" || kind == "f" || kind == "c") {
      nb::object array = numpy.attr("ascontiguousarray")(coeffs_obj, numpy.attr("complex128"));
      const std::size_t size = nb::cast<std::size_t>(array.attr("shape")[0]);
      if (size != expected_size) {
        throw nb::value_error("coefficient count must match label count");
      }
      const auto values = contiguous_span<std::complex<double>>(array, size);
      return {values.begin(), values.end()};
    }
  }

  std::vector<std::complex<double>> coeffs;
  try {
    nb::iterable coeffs_iterable = nb::cast<nb::iterable>(coeffs_obj);
    for (nb::handle item : coeffs_iterable) {
      coeffs.push_back(parse_complex_value(item, "coefficients"));
    }
  } catch (const nb::cast_error&) {
    throw nb::value_error(
        "coeffs must be None, a numeric scalar for one label, or an iterable of numeric values");
  }

  if (coeffs.size() != expected_size) {
    throw nb::value_error("coefficient count must match label count");
  }
  return coeffs;
}

nb::object sequence_item(nb::handle sequence, Py_ssize_t index) {
  PyObject* item = PySequence_GetItem(sequence.ptr(), index);
  if (item == nullptr) {
    throw nb::value_error("failed to read sequence item");
  }
  return nb::steal<nb::object>(item);
}

struct MatrixShape {
  std::size_t rows = 0;
  std::size_t cols = 0;
};

MatrixShape parse_device_commutation_shape(nb::handle shape_obj) {
  if (!PySequence_Check(shape_obj.ptr()) || PyUnicode_Check(shape_obj.ptr())) {
    throw nb::value_error("shape must be a 2-item sequence of non-negative integers");
  }
  if (PySequence_Size(shape_obj.ptr()) != 2) {
    throw nb::value_error("shape must contain exactly 2 dimensions");
  }
  return {
      checked_size_from_python_int(sequence_item(shape_obj, 0), "shape[0]"),
      checked_size_from_python_int(sequence_item(shape_obj, 1), "shape[1]"),
  };
}

std::vector<std::size_t> parse_qubit_indices(nb::handle indices_obj) {
  if (!PySequence_Check(indices_obj.ptr()) || PyUnicode_Check(indices_obj.ptr())) {
    throw nb::value_error("qubit_indices must be a sequence of non-negative integers");
  }
  Py_ssize_t size = PySequence_Size(indices_obj.ptr());
  if (size < 0) {
    throw nb::value_error("failed to read qubit_indices");
  }

  std::vector<std::size_t> indices;
  indices.reserve(static_cast<std::size_t>(size));
  for (Py_ssize_t index = 0; index < size; ++index) {
    nb::object item = sequence_item(indices_obj, index);
    long long qubit = 0;
    try {
      qubit = nb::cast<long long>(item);
    } catch (const nb::cast_error&) {
      throw nb::value_error("qubit_indices must contain integers");
    }
    indices.push_back(checked_size_from_signed(qubit, "qubit index"));
  }
  return indices;
}

std::vector<PauliSum::SparseTerm> parse_sparse_terms(nb::iterable triples_obj) {
  std::vector<PauliSum::SparseTerm> terms;
  for (nb::handle triple_handle : triples_obj) {
    if (!PySequence_Check(triple_handle.ptr()) || PyUnicode_Check(triple_handle.ptr())) {
      throw nb::value_error("each sparse term must be a 3-item sequence");
    }
    Py_ssize_t triple_size = PySequence_Size(triple_handle.ptr());
    if (triple_size != 3) {
      throw nb::value_error("each sparse term must be a 3-item sequence");
    }

    nb::object local_obj = sequence_item(triple_handle, 0);
    nb::object indices_obj = sequence_item(triple_handle, 1);
    nb::object coeff_obj = sequence_item(triple_handle, 2);

    PauliSum::SparseTerm term;
    try {
      term.local_pauli_string = nb::cast<std::string>(local_obj);
    } catch (const nb::cast_error&) {
      throw nb::value_error("sparse local_pauli_string must be a string");
    }
    term.qubit_indices = parse_qubit_indices(indices_obj);
    term.coefficient = parse_complex_value(coeff_obj, "sparse coefficient");
    terms.push_back(std::move(term));
  }
  return terms;
}

void parse_z_counts_mapping(
    nb::handle counts_obj,
    std::vector<std::string>& bitstrings,
    std::vector<double>& counts) {
  if (!PyMapping_Check(counts_obj.ptr())) {
    throw nb::value_error("counts must be a mapping from dense bitstrings to numeric counts");
  }

  PyObject* raw_items = PyMapping_Items(counts_obj.ptr());
  if (raw_items == nullptr) {
    throw nb::python_error();
  }
  nb::object items = nb::steal<nb::object>(raw_items);

  Py_ssize_t item_count = PySequence_Size(items.ptr());
  if (item_count < 0) {
    throw nb::value_error("failed to read counts mapping");
  }

  bitstrings.reserve(static_cast<std::size_t>(item_count));
  counts.reserve(static_cast<std::size_t>(item_count));
  for (Py_ssize_t item_index = 0; item_index < item_count; ++item_index) {
    nb::object item = sequence_item(items, item_index);
    if (!PySequence_Check(item.ptr()) || PyUnicode_Check(item.ptr()) || PySequence_Size(item.ptr()) != 2) {
      throw nb::value_error("counts items must be bitstring/count pairs");
    }

    nb::object bitstring_obj = sequence_item(item, 0);
    nb::object count_obj = sequence_item(item, 1);
    try {
      bitstrings.push_back(nb::cast<std::string>(bitstring_obj));
    } catch (const nb::cast_error&) {
      throw nb::value_error("Z-count bitstrings must be strings");
    }

    const double count = PyFloat_AsDouble(count_obj.ptr());
    if (PyErr_Occurred()) {
      PyErr_Clear();
      throw nb::value_error("Z-count values must be numeric");
    }
    counts.push_back(count);
  }
}

[[noreturn]] void translate_invalid_argument(const std::invalid_argument& error) {
  throw nb::value_error(error.what());
}

AcceleratorBackend parse_backend_selector(nb::object backend_obj) {
  if (backend_obj.is_none()) {
    return AcceleratorBackend::None;
  }
  std::string backend;
  try {
    backend = nb::cast<std::string>(backend_obj);
  } catch (const nb::cast_error&) {
    throw nb::value_error("backend must be None, 'auto', 'cuda', 'hip', or 'metal'");
  }
  try {
    return accelerator_backend_from_name(backend);
  } catch (const std::invalid_argument& error) {
    translate_invalid_argument(error);
  }
}

void ensure_supported_cpu_backend() {
  wolfgang::ensure_cpu_backend_available_from_environment();
}

void ensure_scalar_cpu_operation(std::string_view operation) {
  wolfgang::ensure_cpu_backend_supports_scalar_operation(operation);
}

// Copy a contiguous vector into a new owned 1-D NumPy array of `dtype`.
template <typename T>
nb::object numeric_array_from_vector(const std::vector<T>& values, const char* dtype) {
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::object array = numpy.attr("empty")(nb::make_tuple(values.size()), numpy.attr(dtype));
  if (values.empty()) {
    return array;
  }
  WritablePythonBufferView buffer(array);
  const std::size_t bytes = values.size() * sizeof(T);
  if (buffer.get().len < 0 || static_cast<std::size_t>(buffer.get().len) != bytes) {
    throw std::runtime_error("NumPy array buffer size mismatch");
  }
  std::memcpy(buffer.get().buf, values.data(), bytes);
  return array;
}

nb::object coeff_array_from_vector(const std::vector<std::complex<double>>& coeffs);

// A statevector argument: one 1-D state, or a 2-D C-contiguous batch whose rows
// are states. The buffer export stays held for the duration of the call.
class StatevectorInput {
public:
  explicit StatevectorInput(nb::handle psi_obj) : buffer_(psi_obj) {
    const Py_buffer& view = buffer_.get();
    if (view.ndim != 1 && view.ndim != 2) {
      throw nb::value_error("psi must be 1-dimensional, or a 2-dimensional batch of statevectors");
    }
    if (view.shape == nullptr || view.shape[0] < 0 || (view.ndim == 2 && view.shape[1] < 0)) {
      throw nb::value_error("failed to read psi shape");
    }
    const bool is_complex64 =
        view.format != nullptr && std::strcmp(view.format, "Zf") == 0 && view.itemsize == 8;
    is_complex128_ =
        view.format != nullptr && std::strcmp(view.format, "Zd") == 0 && view.itemsize == 16;
    if (!is_complex64 && !is_complex128_) {
      throw nb::type_error("psi dtype must be complex64 or complex128");
    }
    if (PyBuffer_IsContiguous(&view, 'C') == 0) {
      throw nb::type_error("psi must be C-contiguous");
    }
    batched_ = view.ndim == 2;
    rows_ = batched_ ? static_cast<std::size_t>(view.shape[0]) : 1;
    length_ = static_cast<std::size_t>(view.shape[view.ndim - 1]);
  }

  [[nodiscard]] bool batched() const noexcept { return batched_; }
  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t length() const noexcept { return length_; }

  // Call fn with the row's span typed as complex<double> or complex<float>.
  template <typename Function>
  auto visit_row(std::size_t row, Function&& fn) const {
    const void* data = buffer_.get().buf;
    if (is_complex128_) {
      return fn(std::span<const std::complex<double>>(
          static_cast<const std::complex<double>*>(data) + row * length_, length_));
    }
    return fn(std::span<const std::complex<float>>(
        static_cast<const std::complex<float>*>(data) + row * length_, length_));
  }

  // Call fn with every row as one contiguous span.
  template <typename Function>
  auto visit_all(Function&& fn) const {
    const void* data = buffer_.get().buf;
    const std::size_t count = rows_ * length_;
    if (is_complex128_) {
      return fn(std::span<const std::complex<double>>(
          static_cast<const std::complex<double>*>(data), count));
    }
    return fn(std::span<const std::complex<float>>(
        static_cast<const std::complex<float>*>(data), count));
  }

private:
  PythonBufferView buffer_;
  bool is_complex128_ = false;
  bool batched_ = false;
  std::size_t rows_ = 1;
  std::size_t length_ = 0;
};

// Allocate an owned, writable C-contiguous NumPy array of the given shape and
// return it with a raw pointer to its data. The pointer stays valid while the
// returned object lives.
template <typename T>
std::pair<nb::object, T*> new_numpy_array(std::initializer_list<std::size_t> shape, const char* dtype) {
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::list dims;
  std::size_t count = 1;
  for (std::size_t extent : shape) {
    dims.append(extent);
    count *= extent;
  }
  nb::object array = numpy.attr("empty")(nb::tuple(dims), numpy.attr(dtype));
  if (count == 0) {
    return {array, nullptr};
  }
  WritablePythonBufferView buffer(array);
  return {array, static_cast<T*>(buffer.get().buf)};
}

nb::object expectation_statevector(const PauliSum& op, nb::handle psi_obj) {
  ensure_scalar_cpu_operation("expectation_statevector");
  const StatevectorInput psi(psi_obj);
  try {
    op.validate_statevector_length(psi.length());
    std::vector<std::complex<double>> values(psi.rows());
    cpu_without_gil([&] {
      for (std::size_t row = 0; row < psi.rows(); ++row) {
        values[row] = psi.visit_row(row, [&](auto state) {
          if constexpr (std::is_same_v<typename decltype(state)::element_type, const std::complex<double>>) {
            return op.expectation_statevector_complex128(state);
          } else {
            return op.expectation_statevector_complex64(state);
          }
        });
      }
    });
    if (!psi.batched()) {
      return nb::cast(values.front());
    }
    return coeff_array_from_vector(values);
  } catch (const std::invalid_argument& error) {
    translate_invalid_argument(error);
  }
}

nb::object variance_statevector(const PauliSum& op, nb::handle psi_obj) {
  ensure_scalar_cpu_operation("variance_statevector");
  const StatevectorInput psi(psi_obj);
  try {
    op.validate_statevector_length(psi.length());
    auto [array, out] = new_numpy_array<double>({psi.rows()}, "float64");
    const std::span<double> out_span(out, psi.rows());
    cpu_without_gil([&] {
      psi.visit_all([&](auto states) {
        if constexpr (std::is_same_v<typename decltype(states)::element_type, const std::complex<double>>) {
          op.variance_statevector_batch_complex128(states, psi.rows(), out_span);
        } else {
          op.variance_statevector_batch_complex64(states, psi.rows(), out_span);
        }
      });
    });
    if (!psi.batched()) {
      return nb::cast(out_span.front());
    }
    return array;
  } catch (const std::invalid_argument& error) {
    translate_invalid_argument(error);
  }
}

nb::object apply_statevector(const PauliSum& op, nb::handle psi_obj) {
  ensure_scalar_cpu_operation("apply_statevector");
  const StatevectorInput psi(psi_obj);
  try {
    op.validate_statevector_length(psi.length());
    auto [array, out] = psi.batched()
        ? new_numpy_array<std::complex<double>>({psi.rows(), psi.length()}, "complex128")
        : new_numpy_array<std::complex<double>>({psi.length()}, "complex128");
    const std::span<std::complex<double>> out_span(out, psi.rows() * psi.length());
    cpu_without_gil([&] {
      psi.visit_all([&](auto states) {
        if constexpr (std::is_same_v<typename decltype(states)::element_type, const std::complex<double>>) {
          op.apply_statevector_batch_complex128(states, psi.rows(), out_span);
        } else {
          op.apply_statevector_batch_complex64(states, psi.rows(), out_span);
        }
      });
    });
    return array;
  } catch (const std::invalid_argument& error) {
    translate_invalid_argument(error);
  }
}

bool has_cuda_array_interface(nb::handle object) {
  const int result = PyObject_HasAttrString(object.ptr(), "__cuda_array_interface__");
  if (result < 0) {
    PyErr_Clear();
    return false;
  }
  return result != 0;
}

// The object's __cuda_array_interface__, or TypeError if it has none. Other
// errors raised by the attribute (PyTorch refuses tensors that require grad)
// propagate unchanged, so the caller sees the producer's reason.
[[maybe_unused]] nb::object cuda_array_interface_or_raise(nb::handle object, const char* name) {
  PyObject* interface = PyObject_GetAttrString(object.ptr(), "__cuda_array_interface__");
  if (interface == nullptr) {
    if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
      throw nb::python_error();
    }
    PyErr_Clear();
    throw nb::type_error(
        (std::string(name) + " must expose __cuda_array_interface__ (for example a CuPy array or "
         "a CUDA PyTorch tensor)").c_str());
  }
  return nb::steal<nb::object>(interface);
}

nb::object mapping_get_required(nb::handle mapping, const char* key) {
  PyObject* item = PyMapping_GetItemString(mapping.ptr(), key);
  if (item == nullptr) {
    PyErr_Clear();
    throw nb::type_error(
        (std::string("__cuda_array_interface__ is missing required field ") + key).c_str());
  }
  return nb::steal<nb::object>(item);
}

nb::object mapping_get_optional(nb::handle mapping, const char* key) {
  PyObject* item = PyMapping_GetItemString(mapping.ptr(), key);
  if (item == nullptr) {
    PyErr_Clear();
    return nb::none();
  }
  return nb::steal<nb::object>(item);
}

long long parse_signed_sequence_item(nb::handle sequence, Py_ssize_t index, const char* name) {
  nb::object item = sequence_item(sequence, index);
  if (!PyLong_Check(item.ptr())) {
    throw nb::type_error((std::string(name) + " must contain integers").c_str());
  }
  const long long value = PyLong_AsLongLong(item.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw nb::value_error((std::string(name) + " value is out of range").c_str());
  }
  return value;
}

DeviceStatevectorDtype parse_cuda_statevector_dtype(const std::string& typestr) {
  const char native_order = std::endian::native == std::endian::little ? '<' : '>';
  if (typestr == std::string{native_order} + "c16" || typestr == "=c16") {
    return DeviceStatevectorDtype::Complex128;
  }
  if (typestr == std::string{native_order} + "c8" || typestr == "=c8") {
    return DeviceStatevectorDtype::Complex64;
  }
  throw nb::type_error(
      "CUDA statevector typestr must be native-endian complex64 or complex128");
}

std::size_t cuda_statevector_itemsize(DeviceStatevectorDtype dtype) noexcept {
  return dtype == DeviceStatevectorDtype::Complex128 ? 16 : 8;
}

struct CudaStatevectorView {
  std::uintptr_t pointer = 0;
  DeviceStatevectorDtype dtype = DeviceStatevectorDtype::Complex128;
  std::size_t length = 0;
};

CudaStatevectorView parse_cuda_array_interface_statevector(nb::handle psi_obj) {
  nb::object interface_obj = nb::steal<nb::object>(
      PyObject_GetAttrString(psi_obj.ptr(), "__cuda_array_interface__"));
  if (!PyMapping_Check(interface_obj.ptr())) {
    throw nb::type_error("__cuda_array_interface__ must be a mapping");
  }

  nb::object shape_obj = mapping_get_required(interface_obj, "shape");
  if (!PySequence_Check(shape_obj.ptr()) || PyUnicode_Check(shape_obj.ptr())) {
    throw nb::type_error("__cuda_array_interface__ shape must be a sequence");
  }
  if (PySequence_Size(shape_obj.ptr()) != 1) {
    throw nb::value_error("CUDA statevector must be 1-dimensional");
  }
  const long long raw_length = parse_signed_sequence_item(shape_obj, 0, "shape");
  if (raw_length < 0) {
    throw nb::value_error("CUDA statevector length must be non-negative");
  }

  nb::object typestr_obj = mapping_get_required(interface_obj, "typestr");
  std::string typestr;
  try {
    typestr = nb::cast<std::string>(typestr_obj);
  } catch (const nb::cast_error&) {
    throw nb::type_error("__cuda_array_interface__ typestr must be a string");
  }
  const DeviceStatevectorDtype dtype = parse_cuda_statevector_dtype(typestr);
  const std::size_t itemsize = cuda_statevector_itemsize(dtype);

  nb::object strides_obj = mapping_get_optional(interface_obj, "strides");
  if (!strides_obj.is_none()) {
    if (!PySequence_Check(strides_obj.ptr()) || PyUnicode_Check(strides_obj.ptr()) ||
        PySequence_Size(strides_obj.ptr()) != 1) {
      throw nb::type_error("CUDA statevector strides must be None or a 1-item sequence");
    }
    const long long stride = parse_signed_sequence_item(strides_obj, 0, "strides");
    if (stride != static_cast<long long>(itemsize)) {
      throw nb::type_error("CUDA statevector must be C-contiguous");
    }
  }

  nb::object data_obj = mapping_get_required(interface_obj, "data");
  if (!PySequence_Check(data_obj.ptr()) || PyUnicode_Check(data_obj.ptr()) ||
      PySequence_Size(data_obj.ptr()) < 1) {
    throw nb::type_error("__cuda_array_interface__ data must be a sequence containing a pointer");
  }
  nb::object pointer_obj = sequence_item(data_obj, 0);
  if (!PyLong_Check(pointer_obj.ptr())) {
    throw nb::type_error("__cuda_array_interface__ data pointer must be an integer");
  }
  const unsigned long long pointer = PyLong_AsUnsignedLongLong(pointer_obj.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw nb::value_error("__cuda_array_interface__ data pointer is out of range");
  }
  if (pointer == 0 && raw_length > 0) {
    throw nb::type_error("__cuda_array_interface__ data pointer must be non-null");
  }

  return {
      static_cast<std::uintptr_t>(pointer),
      dtype,
      static_cast<std::size_t>(raw_length),
  };
}

nb::object device_expectation_statevector(const DevicePauliSum& op, nb::handle psi_obj) {
  try {
    if (has_cuda_array_interface(psi_obj)) {
      const CudaStatevectorView view = parse_cuda_array_interface_statevector(psi_obj);
      return nb::cast(op.expectation_statevector_device_pointer(
          view.pointer,
          view.dtype,
          view.length));
    }

    PythonBufferView buffer(psi_obj);
    const Py_buffer& view = buffer.get();

    if (view.ndim != 1) {
      throw nb::value_error("psi must be 1-dimensional");
    }
    if (view.shape == nullptr || view.shape[0] < 0) {
      throw nb::value_error("failed to read psi length");
    }

    const bool is_complex64 =
        view.format != nullptr && std::strcmp(view.format, "Zf") == 0 && view.itemsize == 8;
    const bool is_complex128 =
        view.format != nullptr && std::strcmp(view.format, "Zd") == 0 && view.itemsize == 16;
    if (!is_complex64 && !is_complex128) {
      throw nb::type_error("psi dtype must be complex64 or complex128");
    }
    if (PyBuffer_IsContiguous(&view, 'C') == 0) {
      throw nb::type_error("psi must be C-contiguous");
    }

    const std::size_t size = static_cast<std::size_t>(view.shape[0]);
    if (is_complex128) {
      return nb::cast(op.expectation_statevector_complex128(
          std::span<const std::complex<double>>(
              static_cast<const std::complex<double>*>(view.buf),
              size)));
    }
    return nb::cast(op.expectation_statevector_complex64(
        std::span<const std::complex<float>>(
            static_cast<const std::complex<float>*>(view.buf),
            size)));
  } catch (const std::invalid_argument& error) {
    translate_invalid_argument(error);
  }
}

nb::object coeff_array_from_vector(const std::vector<std::complex<double>>& coeffs) {
  static_assert(sizeof(std::complex<double>) == 2 * sizeof(double));
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::object array = numpy.attr("empty")(nb::make_tuple(coeffs.size()), numpy.attr("complex128"));
  if (coeffs.empty()) {
    return array;
  }
  WritablePythonBufferView buffer(array);
  const std::size_t bytes = coeffs.size() * sizeof(std::complex<double>);
  if (buffer.get().len < 0 || static_cast<std::size_t>(buffer.get().len) != bytes) {
    throw std::runtime_error("NumPy complex128 array buffer size mismatch");
  }
  // std::complex<double> uses adjacent real/imaginary doubles. One owning NumPy
  // allocation avoids a Python list and a boxed complex object per coefficient.
  std::memcpy(buffer.get().buf, coeffs.data(), bytes);
  return array;
}

// Packed-array boundary. Inputs are normalized through NumPy so non-contiguous
// arrays are accepted (the operator always copies), but the word dtype is never
// cast: a signed or float array could reinterpret bits silently.
struct PackedArrayInput {
  nb::object x;
  nb::object z;
  nb::object coeffs;
};

PackedArrayInput normalize_packed_arrays(
    nb::handle x_obj,
    nb::handle z_obj,
    nb::handle coeffs_obj,
    std::size_t words) {
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::object uint64 = numpy.attr("dtype")("uint64");
  auto words_array = [&](nb::handle obj, const char* name) {
    nb::object array = numpy.attr("asarray")(obj);
    if (!array.attr("dtype").equal(uint64)) {
      throw nb::type_error(
          (std::string(name) + " must have dtype uint64 in native byte order").c_str());
    }
    if (nb::cast<int>(array.attr("ndim")) != 2) {
      throw nb::value_error((std::string(name) + " must have shape (num_terms, words)").c_str());
    }
    return numpy.attr("ascontiguousarray")(array);
  };
  PackedArrayInput input;
  input.x = words_array(x_obj, "x");
  input.z = words_array(z_obj, "z");

  nb::object coeffs = numpy.attr("asarray")(coeffs_obj);
  const std::string kind = nb::cast<std::string>(coeffs.attr("dtype").attr("kind"));
  if (kind != "b" && kind != "i" && kind != "u" && kind != "f" && kind != "c") {
    throw nb::type_error("coeffs must be a numeric array");
  }
  if (nb::cast<int>(coeffs.attr("ndim")) != 1) {
    throw nb::value_error("coeffs must be 1-dimensional");
  }
  input.coeffs = numpy.attr("ascontiguousarray")(coeffs, numpy.attr("complex128"));

  const std::size_t num_terms = nb::cast<std::size_t>(input.coeffs.attr("shape")[0]);
  for (const nb::object* array : {&input.x, &input.z}) {
    nb::object shape = array->attr("shape");
    if (nb::cast<std::size_t>(shape[0]) != num_terms || nb::cast<std::size_t>(shape[1]) != words) {
      throw nb::value_error(
          "x and z must have shape (len(coeffs), ceil(num_qubits / 64))");
    }
  }
  return input;
}

// Read-only NumPy views over the operator's own buffers. The view holds a
// reference to the Python PauliSum, and PauliSum is immutable, so the memory
// stays valid and unchanged for the view's lifetime.
template <typename T>
nb::object readonly_view(nb::handle owner, const T* data, std::initializer_list<std::size_t> shape) {
  static const T empty_storage{};
  std::size_t count = 1;
  for (std::size_t extent : shape) {
    count *= extent;
  }
  const T* pointer = count == 0 ? &empty_storage : data;
  return nb::cast(nb::ndarray<nb::numpy, const T>(
      pointer, shape.size(), shape.begin(), owner));
}

nb::object bool_array_from_vector(const std::vector<std::uint8_t>& values) {
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::object array = numpy.attr("empty")(nb::make_tuple(values.size()), numpy.attr("bool_"));
  if (values.empty()) {
    return array;
  }

  WritablePythonBufferView buffer(array);
  const Py_buffer& view = buffer.get();
  if (view.len != static_cast<Py_ssize_t>(values.size())) {
    throw std::runtime_error("NumPy bool array buffer size mismatch");
  }
  std::memcpy(view.buf, values.data(), values.size());
  return array;
}

nb::object bool_matrix_from_device_commutation(const DeviceCommutationMatrix& matrix) {
  nb::object array = bool_array_from_vector(matrix.to_host());
  return array.attr("reshape")(nb::make_tuple(matrix.rows(), matrix.cols()));
}

nb::object uint64_array_from_vector(const std::vector<std::uint64_t>& values) {
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::object array = numpy.attr("empty")(nb::make_tuple(values.size()), numpy.attr("uint64"));
  if (values.empty()) {
    return array;
  }

  WritablePythonBufferView buffer(array);
  const Py_buffer& view = buffer.get();
  const std::size_t byte_count = values.size() * sizeof(std::uint64_t);
  if (view.len != static_cast<Py_ssize_t>(byte_count)) {
    throw std::runtime_error("NumPy uint64 array buffer size mismatch");
  }
  std::memcpy(view.buf, values.data(), byte_count);
  return array;
}

nb::object count_commuting_device_matrix(
    const DeviceCommutationMatrix& matrix,
    nb::object axis_obj) {
  try {
    if (axis_obj.is_none()) {
      return nb::steal<nb::object>(PyLong_FromUnsignedLongLong(
          static_cast<unsigned long long>(matrix.count_commuting())));
    }

    long long axis = 0;
    try {
      axis = nb::cast<long long>(axis_obj);
    } catch (const nb::cast_error&) {
      throw nb::value_error("axis must be None, 0, or 1");
    }
    if (axis == 0) {
      return uint64_array_from_vector(matrix.count_commuting_cols());
    }
    if (axis == 1) {
      return uint64_array_from_vector(matrix.count_commuting_rows());
    }
    throw nb::value_error("axis must be None, 0, or 1");
  } catch (const std::invalid_argument& error) {
    translate_invalid_argument(error);
  }
}

nb::object conflict_degrees_device_matrix(
    const DeviceCommutationMatrix& matrix,
    nb::object axis_obj) {
  try {
    if (axis_obj.is_none()) {
      const auto entries = static_cast<unsigned long long>(matrix.num_entries());
      const auto commuting = static_cast<unsigned long long>(matrix.count_commuting());
      return nb::steal<nb::object>(PyLong_FromUnsignedLongLong(entries - commuting));
    }

    long long axis = 0;
    try {
      axis = nb::cast<long long>(axis_obj);
    } catch (const nb::cast_error&) {
      throw nb::value_error("axis must be None, 0, or 1");
    }
    if (axis == 0) {
      std::vector<std::uint64_t> counts = matrix.count_commuting_cols();
      const std::uint64_t row_count = static_cast<std::uint64_t>(matrix.rows());
      for (std::uint64_t& value : counts) {
        value = row_count - value;
      }
      return uint64_array_from_vector(counts);
    }
    if (axis == 1) {
      std::vector<std::uint64_t> counts = matrix.count_commuting_rows();
      const std::uint64_t col_count = static_cast<std::uint64_t>(matrix.cols());
      for (std::uint64_t& value : counts) {
        value = col_count - value;
      }
      return uint64_array_from_vector(counts);
    }
    throw nb::value_error("axis must be None, 0, or 1");
  } catch (const std::invalid_argument& error) {
    translate_invalid_argument(error);
  }
}

void device_commutation_dlpack_deleter(DLManagedTensor* managed) {
  if (managed == nullptr) {
    return;
  }
  auto* context = static_cast<DeviceCommutationDlpackContext*>(managed->manager_ctx);
  delete context;
  delete managed;
}

void device_commutation_versioned_dlpack_deleter(DLManagedTensorVersioned* managed) {
  if (managed == nullptr) {
    return;
  }
  auto* context = static_cast<DeviceCommutationDlpackContext*>(managed->manager_ctx);
  delete context;
  delete managed;
}

void device_commutation_dlpack_capsule_destructor(PyObject* capsule) {
  const char* name = PyCapsule_GetName(capsule);
  if (name != nullptr && std::strcmp(name, "used_dltensor") == 0) {
    return;
  }
  if (name == nullptr || std::strcmp(name, "dltensor") != 0) {
    PyErr_Clear();
    return;
  }

  auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
  if (managed != nullptr && managed->deleter != nullptr) {
    managed->deleter(managed);
  } else {
    PyErr_Clear();
  }
}

void device_commutation_versioned_dlpack_capsule_destructor(PyObject* capsule) {
  const char* name = PyCapsule_GetName(capsule);
  if (name != nullptr && std::strcmp(name, "used_dltensor_versioned") == 0) {
    return;
  }
  if (name == nullptr || std::strcmp(name, "dltensor_versioned") != 0) {
    PyErr_Clear();
    return;
  }

  auto* managed = static_cast<DLManagedTensorVersioned*>(
      PyCapsule_GetPointer(capsule, "dltensor_versioned"));
  if (managed != nullptr && managed->deleter != nullptr) {
    managed->deleter(managed);
  } else {
    PyErr_Clear();
  }
}

void validate_dlpack_arguments(
    nb::object stream_obj,
    nb::object max_version_obj,
    nb::object copy_obj,
    const char* type_name = "DeviceCommutationMatrix") {
  if (!copy_obj.is_none()) {
    if (PyObject_IsTrue(copy_obj.ptr()) == 1) {
      throw_buffer_error((std::string(type_name) + ".__dlpack__ does not support copy=True").c_str());
    }
    PyErr_Clear();
  }
  if (!stream_obj.is_none()) {
    long long stream = 0;
    try {
      stream = nb::cast<long long>(stream_obj);
    } catch (const nb::cast_error&) {
      throw nb::type_error("stream must be None or an integer accelerator stream token");
    }
    if (stream == 0) {
      throw nb::value_error("stream=0 is ambiguous in the Python DLPack protocol");
    }
    if (stream < 0) {
      throw nb::value_error("stream must be non-negative");
    }
  }
}

std::pair<std::uint32_t, std::uint32_t> parse_dlpack_max_version(nb::object max_version_obj) {
  if (max_version_obj.is_none()) {
    return {0, 0};
  }
  try {
    nb::tuple tuple = nb::cast<nb::tuple>(max_version_obj);
    if (tuple.size() != 2) {
      throw nb::value_error("max_version must be None or a (major, minor) tuple");
    }
    long long major = nb::cast<long long>(tuple[0]);
    long long minor = nb::cast<long long>(tuple[1]);
    if (major < 0 || minor < 0 ||
        static_cast<unsigned long long>(major) > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<unsigned long long>(minor) > std::numeric_limits<std::uint32_t>::max()) {
      throw nb::value_error("max_version components must be non-negative uint32 values");
    }
    return {static_cast<std::uint32_t>(major), static_cast<std::uint32_t>(minor)};
  } catch (const nb::cast_error&) {
    throw nb::value_error("max_version must be None or a (major, minor) tuple");
  }
}

nb::object device_commutation_matrix_dlpack(
    const DeviceCommutationMatrix& matrix,
    nb::object stream_obj,
    nb::object max_version_obj,
    nb::object copy_obj) {
  validate_dlpack_arguments(stream_obj, max_version_obj, copy_obj);
  const auto [major, minor] = parse_dlpack_max_version(max_version_obj);
  const auto negotiated = major == 0 && minor == 0 && max_version_obj.is_none()
      ? std::optional<DLPackVersion>{}
      : detail::negotiate_dlpack_version(DLPackVersion{major, minor});
  if (!negotiated.has_value()) {
    throw_buffer_error(
        "DeviceCommutationMatrix.__dlpack__ requires max_version >= (1, 0) "
        "so the read-only DLPack contract can be represented");
  }
  const auto pointer = matrix.data_pointer_for_dlpack();

  auto context = std::make_unique<DeviceCommutationDlpackContext>();
  context->shape = std::make_unique<std::int64_t[]>(2);
  context->shape[0] = static_cast<std::int64_t>(matrix.rows());
  context->shape[1] = static_cast<std::int64_t>(matrix.cols());
  context->strides = std::make_unique<std::int64_t[]>(2);
  context->strides[0] = static_cast<std::int64_t>(matrix.cols());
  context->strides[1] = 1;

  nb::object owner = nb::cast(&matrix, nb::rv_policy::reference);
  Py_INCREF(owner.ptr());
  context->owner = owner.ptr();

  auto managed = std::make_unique<DLManagedTensorVersioned>();
  managed->version = *negotiated;
  managed->manager_ctx = context.get();
  managed->deleter = &device_commutation_versioned_dlpack_deleter;
  managed->flags = DLPACK_FLAG_BITMASK_READ_ONLY;
  managed->dl_tensor.data = reinterpret_cast<void*>(pointer);
  managed->dl_tensor.device = DLDevice{
      static_cast<DLDeviceType>(matrix.dlpack_device_type()),
      matrix.device()};
  managed->dl_tensor.ndim = 2;
  managed->dl_tensor.dtype = DLDataType{static_cast<std::uint8_t>(kDLUInt), 8, 1};
  managed->dl_tensor.shape = context->shape.get();
  managed->dl_tensor.strides = context->strides.get();
  managed->dl_tensor.byte_offset = 0;

  PyObject* capsule = PyCapsule_New(
      managed.get(),
      "dltensor_versioned",
      &device_commutation_versioned_dlpack_capsule_destructor);
  if (capsule == nullptr) {
    throw nb::python_error();
  }
  context.release();
  managed.release();
  return nb::steal<nb::object>(capsule);
}

nb::tuple device_commutation_matrix_dlpack_device(const DeviceCommutationMatrix& matrix) {
  return nb::make_tuple(matrix.dlpack_device_type(), matrix.device());
}

nb::object bool_array_from_device_commutation(
    const DevicePauliSum& lhs,
    const DevicePauliSum& rhs,
    std::size_t entries,
    std::size_t max_commutation_matrix_entries) {
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::object array = numpy.attr("empty")(nb::make_tuple(entries), numpy.attr("bool_"));
  if (entries == 0) {
    return array;
  }

  WritablePythonBufferView buffer(array);
  const Py_buffer& view = buffer.get();
  if (view.len != static_cast<Py_ssize_t>(entries)) {
    throw std::runtime_error("NumPy bool array buffer size mismatch");
  }
  lhs.commutes_with_into(
      rhs,
      std::span<std::uint8_t>(static_cast<std::uint8_t*>(view.buf), entries),
      max_commutation_matrix_entries);
  return array;
}

std::span<std::uint8_t> writable_flat_bool_span(
    const Py_buffer& view,
    std::size_t expected_entries) {
  const bool is_bool =
      view.format != nullptr && std::strcmp(view.format, "?") == 0 && view.itemsize == 1;
  if (!is_bool) {
    throw nb::type_error("commutes_with_into output dtype must be bool");
  }
  if (view.ndim != 1) {
    throw nb::value_error("commutes_with_into output must be a 1-dimensional array");
  }
  if (PyBuffer_IsContiguous(&view, 'C') == 0) {
    throw nb::type_error("commutes_with_into output must be C-contiguous");
  }
  if (view.shape == nullptr || view.shape[0] < 0) {
    throw nb::value_error("failed to read commutes_with_into output length");
  }
  const std::size_t entries = static_cast<std::size_t>(view.shape[0]);
  if (entries != expected_entries) {
    throw nb::value_error("CUDA commutes_with_into output buffer size does not match entry count");
  }
  return std::span<std::uint8_t>(static_cast<std::uint8_t*>(view.buf), entries);
}

nb::dict cuda_device_info_to_dict(const CudaDeviceInfo& device) {
  nb::dict item;
  item["ordinal"] = device.ordinal;
  item["name"] = device.name;
  item["compute_capability"] = nb::make_tuple(
      device.compute_capability_major,
      device.compute_capability_minor);
  item["total_memory_bytes"] = device.total_memory_bytes;
  return item;
}

nb::dict cuda_status_to_dict(const CudaStatus& status) {
  nb::dict info;
  info["built"] = status.built;
  info["runtime_available"] = status.runtime_available;
  info["device_count"] = status.device_count;
  info["skip_reason"] = status.skip_reason;
  info["runtime_version"] = status.runtime_version;
  info["driver_version"] = status.driver_version;

  nb::list devices;
  for (const CudaDeviceInfo& device : status.devices) {
    devices.append(cuda_device_info_to_dict(device));
  }
  info["devices"] = devices;
  return info;
}

nb::dict hip_device_info_to_dict(const HipDeviceInfo& device) {
  nb::dict item;
  item["ordinal"] = device.ordinal;
  item["name"] = device.name;
  item["gfx_target"] = device.gfx_target;
  item["total_memory_bytes"] = device.total_memory_bytes;
  return item;
}

nb::dict hip_status_to_dict(const HipStatus& status) {
  nb::dict info;
  info["built"] = status.built;
  info["runtime_available"] = status.runtime_available;
  info["device_count"] = status.device_count;
  info["skip_reason"] = status.skip_reason;
  info["runtime_version"] = status.runtime_version;
  info["driver_version"] = status.driver_version;
  info["toolkit_version"] = status.toolkit_version;

  nb::list devices;
  for (const HipDeviceInfo& device : status.devices) {
    devices.append(hip_device_info_to_dict(device));
  }
  info["devices"] = devices;
  return info;
}

nb::dict metal_device_info_to_dict(const MetalDeviceInfo& device) {
  nb::dict item;
  item["ordinal"] = device.ordinal;
  item["name"] = device.name;
  item["registry_id"] = device.registry_id;
  item["recommended_max_working_set_size"] = device.recommended_max_working_set_size;
  item["capability_summary"] = device.capability_summary;
  item["low_power"] = device.low_power;
  item["headless"] = device.headless;
  item["removable"] = device.removable;
  item["unified_memory"] = device.unified_memory;
  return item;
}

nb::dict metal_status_to_dict(const MetalStatus& status) {
  nb::dict info;
  info["built"] = status.built;
  info["runtime_available"] = status.runtime_available;
  info["device_count"] = status.device_count;
  info["skip_reason"] = status.skip_reason;
  info["macos_version"] = status.macos_version;
  info["xcode_or_clt_version"] = status.xcode_or_clt_version;
  info["metal_device_name"] = status.metal_device_name;
  info["storage_mode"] = status.storage_mode;
  info["capability_summary"] = status.capability_summary;

  nb::list devices;
  for (const MetalDeviceInfo& device : status.devices) {
    devices.append(metal_device_info_to_dict(device));
  }
  info["devices"] = devices;
  return info;
}

nb::dict accelerator_status_to_dict() {
  const CudaStatus cuda_status = DevicePauliSum::cuda_status();
  const HipStatus hip_status = DevicePauliSum::hip_status();
  const MetalStatus metal_status = DevicePauliSum::metal_status();
  nb::dict info;
  info["cuda"] = cuda_status_to_dict(cuda_status);
  info["hip"] = hip_status_to_dict(hip_status);
  info["metal"] = metal_status_to_dict(metal_status);

  nb::list compiled_backends;
  compiled_backends.append("cpu");
  nb::list compiled_accelerator_backends;
  if (cuda_status.built) {
    compiled_backends.append("cuda");
    compiled_accelerator_backends.append("cuda");
  }
  if (hip_status.built) {
    compiled_backends.append("hip");
    compiled_accelerator_backends.append("hip");
  }
  if (metal_status.built) {
    compiled_backends.append("metal");
    compiled_accelerator_backends.append("metal");
  }
  info["compiled_backends"] = compiled_backends;
  info["compiled_accelerator_backends"] = compiled_accelerator_backends;

  nb::list available_backends;
  available_backends.append("cpu");
  nb::list available_accelerator_backends;
  if (cuda_status.runtime_available) {
    available_backends.append("cuda");
    available_accelerator_backends.append("cuda");
  }
  if (hip_status.runtime_available) {
    available_backends.append("hip");
    available_accelerator_backends.append("hip");
  }
  if (metal_status.runtime_available) {
    available_backends.append("metal");
    available_accelerator_backends.append("metal");
  }
  info["available_backends"] = available_backends;
  info["available_accelerator_backends"] = available_accelerator_backends;

  if (cuda_status.runtime_available && !hip_status.runtime_available &&
      !metal_status.runtime_available) {
    info["active_backend"] = "cuda";
  } else if (hip_status.runtime_available && !cuda_status.runtime_available &&
             !metal_status.runtime_available) {
    info["active_backend"] = "hip";
  } else if (metal_status.runtime_available && !cuda_status.runtime_available &&
             !hip_status.runtime_available) {
    info["active_backend"] = "metal";
  } else {
    info["active_backend"] = "none";
  }
  return info;
}

// A read-only view of one DevicePauliSum device buffer. It keeps the owning
// DevicePauliSum alive, and every DLPack capsule it exports holds its own
// reference, so the buffer outlives every consumer.
struct DeviceArrayView {
  nb::object owner;
  std::uintptr_t pointer = 0;
  std::vector<std::int64_t> shape;
  DLDataType dtype{};
  const char* typestr = "";
  const char* dtype_name = "";
  int device_type = 0;
  int device = 0;
};

nb::tuple device_array_shape(const DeviceArrayView& view) {
  nb::list items;
  for (const std::int64_t extent : view.shape) {
    items.append(extent);
  }
  return nb::tuple(items);
}

nb::object device_array_dlpack(
    const DeviceArrayView& view,
    nb::object stream_obj,
    nb::object max_version_obj,
    nb::object dl_device_obj,
    nb::object copy_obj) {
  validate_dlpack_arguments(stream_obj, max_version_obj, copy_obj, "DeviceArrayView");
  // A consumer may name the device it wants (array API dl_device); only the
  // buffer's own device can be served without a copy.
  if (!dl_device_obj.is_none()) {
    bool matches = false;
    try {
      const nb::tuple requested = nb::cast<nb::tuple>(dl_device_obj);
      matches = requested.size() == 2 && nb::cast<int>(requested[0]) == view.device_type &&
          nb::cast<int>(requested[1]) == view.device;
    } catch (const nb::cast_error&) {
      throw nb::type_error("dl_device must be None or a (device_type, device_id) tuple");
    }
    if (!matches) {
      throw_buffer_error("DeviceArrayView.__dlpack__ cannot copy to another device");
    }
  }
  const auto [major, minor] = parse_dlpack_max_version(max_version_obj);
  const auto negotiated = major == 0 && minor == 0 && max_version_obj.is_none()
      ? std::optional<DLPackVersion>{}
      : detail::negotiate_dlpack_version(DLPackVersion{major, minor});
  if (!negotiated.has_value()) {
    throw_buffer_error(
        "DeviceArrayView.__dlpack__ requires max_version >= (1, 0) "
        "so the read-only DLPack contract can be represented");
  }
  const std::size_t ndim = view.shape.size();
  auto context = std::make_unique<DeviceCommutationDlpackContext>();
  context->shape = std::make_unique<std::int64_t[]>(ndim);
  context->strides = std::make_unique<std::int64_t[]>(ndim);
  std::int64_t stride = 1;
  for (std::size_t axis = ndim; axis-- > 0;) {
    context->shape[axis] = view.shape[axis];
    context->strides[axis] = stride;
    stride *= view.shape[axis];
  }
  Py_INCREF(view.owner.ptr());
  context->owner = view.owner.ptr();

  auto managed = std::make_unique<DLManagedTensorVersioned>();
  managed->version = *negotiated;
  managed->manager_ctx = context.get();
  managed->deleter = &device_commutation_versioned_dlpack_deleter;
  managed->flags = DLPACK_FLAG_BITMASK_READ_ONLY;
  managed->dl_tensor.data = reinterpret_cast<void*>(view.pointer);
  managed->dl_tensor.device = DLDevice{static_cast<DLDeviceType>(view.device_type), view.device};
  managed->dl_tensor.ndim = static_cast<std::int32_t>(ndim);
  managed->dl_tensor.dtype = view.dtype;
  managed->dl_tensor.shape = context->shape.get();
  managed->dl_tensor.strides = context->strides.get();
  managed->dl_tensor.byte_offset = 0;

  PyObject* capsule = PyCapsule_New(
      managed.get(),
      "dltensor_versioned",
      &device_commutation_versioned_dlpack_capsule_destructor);
  if (capsule == nullptr) {
    throw nb::python_error();
  }
  context.release();
  managed.release();
  return nb::steal<nb::object>(capsule);
}

nb::dict device_array_cuda_array_interface(const DeviceArrayView& view) {
  nb::dict interface;
  interface["shape"] = device_array_shape(view);
  interface["typestr"] = view.typestr;
  interface["data"] = nb::make_tuple(static_cast<unsigned long long>(view.pointer), true);
  interface["version"] = 3;
  interface["strides"] = nb::none();
  // Wolfgang operations synchronize before returning, so the data is ready on
  // any stream; 1 is the legacy default stream.
  interface["stream"] = 1;
  return interface;
}

nb::tuple device_pauli_sum_device_arrays(nb::pointer_and_handle<DevicePauliSum> self) {
  const DevicePauliSum& op = *self.p;
  const DeviceArrayPointers pointers = op.export_device_arrays();
  const auto terms = static_cast<std::int64_t>(op.num_terms());
  const auto words = static_cast<std::int64_t>(op.words());
  const nb::object owner = nb::borrow(self.h);
  const char native_order = std::endian::native == std::endian::little ? '<' : '>';
  static const std::string u8_typestr = std::string{native_order} + "u8";
  static const std::string c16_typestr = std::string{native_order} + "c16";
  const DLDataType uint64_dtype{static_cast<std::uint8_t>(kDLUInt), 64, 1};
  const DLDataType complex128_dtype{static_cast<std::uint8_t>(kDLComplex), 128, 1};
  DeviceArrayView x{owner, pointers.x, {terms, words}, uint64_dtype, u8_typestr.c_str(), "uint64",
                    pointers.dlpack_device_type, pointers.device};
  DeviceArrayView z{owner, pointers.z, {terms, words}, uint64_dtype, u8_typestr.c_str(), "uint64",
                    pointers.dlpack_device_type, pointers.device};
  DeviceArrayView coeffs{owner, pointers.coeffs, {terms}, complex128_dtype, c16_typestr.c_str(), "complex128",
                         pointers.dlpack_device_type, pointers.device};
  return nb::make_tuple(nb::cast(std::move(x)), nb::cast(std::move(z)), nb::cast(std::move(coeffs)));
}

// A device array received through __cuda_array_interface__ with the expected
// typestr and shape, C-contiguous.
[[maybe_unused]] std::uintptr_t parse_cuda_array_interface_input(
    nb::handle object,
    const char* name,
    const std::string& expected_typestr,
    std::initializer_list<long long> expected_shape,
    std::size_t itemsize) {
  nb::object interface_obj = cuda_array_interface_or_raise(object, name);
  if (!PyMapping_Check(interface_obj.ptr())) {
    throw nb::type_error("__cuda_array_interface__ must be a mapping");
  }
  nb::object typestr_obj = mapping_get_required(interface_obj, "typestr");
  std::string typestr;
  try {
    typestr = nb::cast<std::string>(typestr_obj);
  } catch (const nb::cast_error&) {
    throw nb::type_error("__cuda_array_interface__ typestr must be a string");
  }
  if (typestr != expected_typestr && typestr != "=" + expected_typestr.substr(1)) {
    throw nb::type_error((std::string(name) + " must have dtype " + expected_typestr).c_str());
  }
  nb::object shape_obj = mapping_get_required(interface_obj, "shape");
  if (!PySequence_Check(shape_obj.ptr()) || PyUnicode_Check(shape_obj.ptr()) ||
      PySequence_Size(shape_obj.ptr()) != static_cast<Py_ssize_t>(expected_shape.size())) {
    throw nb::value_error((std::string(name) + " has the wrong number of dimensions").c_str());
  }
  Py_ssize_t axis = 0;
  for (const long long extent : expected_shape) {
    if (parse_signed_sequence_item(shape_obj, axis, "shape") != extent) {
      throw nb::value_error((std::string(name) + " has the wrong shape").c_str());
    }
    ++axis;
  }
  nb::object mask_obj = mapping_get_optional(interface_obj, "mask");
  if (!mask_obj.is_none()) {
    throw nb::type_error((std::string(name) + " must not be a masked array").c_str());
  }
  // Strides are irrelevant for an empty array, whatever the producer reports.
  const bool empty = std::find(expected_shape.begin(), expected_shape.end(), 0LL) != expected_shape.end();
  nb::object strides_obj = mapping_get_optional(interface_obj, "strides");
  if (!strides_obj.is_none() && !empty) {
    if (!PySequence_Check(strides_obj.ptr()) || PyUnicode_Check(strides_obj.ptr()) ||
        PySequence_Size(strides_obj.ptr()) != static_cast<Py_ssize_t>(expected_shape.size())) {
      throw nb::type_error((std::string(name) + " strides must match its shape").c_str());
    }
    long long expected_stride = static_cast<long long>(itemsize);
    for (Py_ssize_t index = static_cast<Py_ssize_t>(expected_shape.size()); index-- > 0;) {
      const long long extent = *(expected_shape.begin() + index);
      if (extent > 1 && parse_signed_sequence_item(strides_obj, index, "strides") != expected_stride) {
        throw nb::type_error((std::string(name) + " must be C-contiguous").c_str());
      }
      expected_stride *= extent;
    }
  }
  nb::object data_obj = mapping_get_required(interface_obj, "data");
  if (!PySequence_Check(data_obj.ptr()) || PyUnicode_Check(data_obj.ptr()) ||
      PySequence_Size(data_obj.ptr()) < 1) {
    throw nb::type_error("__cuda_array_interface__ data must be a sequence containing a pointer");
  }
  nb::object pointer_obj = sequence_item(data_obj, 0);
  if (!PyLong_Check(pointer_obj.ptr())) {
    throw nb::type_error("__cuda_array_interface__ data pointer must be an integer");
  }
  const unsigned long long pointer = PyLong_AsUnsignedLongLong(pointer_obj.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw nb::value_error("__cuda_array_interface__ data pointer is out of range");
  }
  return static_cast<std::uintptr_t>(pointer);
}

DevicePauliSum device_pauli_sum_from_device_arrays(
    nb::handle x_obj, nb::handle z_obj, nb::handle coeffs_obj, std::size_t num_qubits, int device) {
#if !WOLFGANG_BUILD_CUDA_ENABLED
  (void)x_obj;
  (void)z_obj;
  (void)coeffs_obj;
  (void)num_qubits;
  (void)device;
  throw std::runtime_error(
      "DevicePauliSum.from_device_arrays requires a CUDA build (WOLFGANG_ENABLE_CUDA=ON); "
      "HIP and Metal builds do not import device arrays yet");
#else
  if (num_qubits > std::numeric_limits<std::size_t>::max() - 63) {
    throw nb::value_error("num_qubits is too large");
  }
  nb::object coeff_interface = cuda_array_interface_or_raise(coeffs_obj, "coeffs");
  nb::object coeff_shape = mapping_get_required(coeff_interface, "shape");
  if (!PySequence_Check(coeff_shape.ptr()) || PySequence_Size(coeff_shape.ptr()) != 1) {
    throw nb::value_error("coeffs must be 1-dimensional");
  }
  const long long num_terms = parse_signed_sequence_item(coeff_shape, 0, "shape");
  if (num_terms < 0) {
    throw nb::value_error("coeffs length must be non-negative");
  }
  const long long words = static_cast<long long>((num_qubits + 63) / 64);
  const char native_order = std::endian::native == std::endian::little ? '<' : '>';
  const std::string u8 = std::string{native_order} + "u8";
  const std::string c16 = std::string{native_order} + "c16";
  const std::uintptr_t coeffs = parse_cuda_array_interface_input(coeffs_obj, "coeffs", c16, {num_terms}, 16);
  const std::uintptr_t x = parse_cuda_array_interface_input(x_obj, "x", u8, {num_terms, words}, 8);
  const std::uintptr_t z = parse_cuda_array_interface_input(z_obj, "z", u8, {num_terms, words}, 8);
  try {
    return DevicePauliSum::from_device_arrays(
        num_qubits, static_cast<std::size_t>(num_terms), x, z, coeffs, AcceleratorBackend::Cuda, device);
  } catch (const std::invalid_argument& error) {
    translate_invalid_argument(error);
  }
#endif
}


nb::dict device_commutation_matrix_cuda_array_interface(
    const DeviceCommutationMatrix& matrix) {
  nb::dict interface;
  interface["shape"] = nb::make_tuple(matrix.rows(), matrix.cols());
  interface["typestr"] = "|u1";
  interface["data"] = nb::make_tuple(
      static_cast<unsigned long long>(matrix.data_pointer_for_cuda_array_interface()),
      false);
  interface["version"] = 3;
  interface["strides"] = nb::none();
  interface["stream"] = 1;
  return interface;
}


}  // namespace

void bind_pauli_sum(nb::module_& module) {
  nb::class_<DeviceCommutationMatrix>(
      module,
      "DeviceCommutationMatrix",
      "Owning accelerator device-resident dense uint8 commutation matrix.")
      .def_static(
          "empty",
          [](nb::handle shape_obj, int device, nb::object backend_obj) {
            const MatrixShape shape = parse_device_commutation_shape(shape_obj);
            try {
              return DeviceCommutationMatrix::empty(
                  shape.rows,
                  shape.cols,
                  parse_backend_selector(std::move(backend_obj)),
                  device);
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("shape"),
          nb::arg("device") = 0,
          nb::arg("backend").none() = nb::none(),
          "Allocate an empty device-resident dense uint8 commutation matrix.\n\n"
          "backend may be None, 'auto', 'cuda', 'hip', or 'metal'. None and 'auto' "
          "preserve single-backend compatibility and raise an ambiguity error "
          "when a future dual-backend build cannot choose deterministically.")
      .def_prop_ro(
          "shape",
          [](const DeviceCommutationMatrix& matrix) {
            return nb::make_tuple(matrix.rows(), matrix.cols());
          },
          "Matrix shape as (rows, cols).")
      .def_prop_ro(
          "device",
          &DeviceCommutationMatrix::device,
          "Accelerator device ordinal that owns the matrix allocation.")
      .def_prop_ro(
          "backend",
          &DeviceCommutationMatrix::backend,
          "Compiled accelerator backend that owns the matrix buffer: 'cuda', 'hip', or 'metal'.")
      .def_prop_ro(
          "dtype",
          [](const DeviceCommutationMatrix&) { return "uint8"; },
          "Device buffer dtype for the owned dense commutation matrix.")
      .def_prop_ro(
          "num_entries",
          &DeviceCommutationMatrix::num_entries,
          "Number of dense commutation flags in the matrix.")
      .def(
          "to_host",
          [](const DeviceCommutationMatrix& matrix) {
            try {
              return bool_matrix_from_device_commutation(matrix);
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          "Copy the device uint8 flags to a host NumPy bool matrix.")
      .def(
          "count_commuting",
          &count_commuting_device_matrix,
          nb::arg("axis") = nb::none(),
          "Count commuting entries using the backend consumer.\n\n"
          "CUDA/HIP reduce on device and may finish partial sums on the host. Metal scans "
          "shared memory on the CPU by default. "
          "axis=None returns a Python int total. axis=0 returns NumPy uint64 "
          "column counts. axis=1 returns NumPy uint64 row counts. The method "
          "synchronizes before returning, matching Wolfgang's public accelerator "
          "method semantics.")
      .def(
          "conflict_degrees",
          &conflict_degrees_device_matrix,
          nb::arg("axis") = nb::none(),
          "Count anti-commuting entries using the backend consumer.\n\n"
          "CUDA/HIP reduce on device and may finish partial sums on the host. Metal scans "
          "shared memory on the CPU by default. "
          "axis=None returns a Python int total. axis=0 returns NumPy uint64 "
          "column conflict counts. axis=1 returns NumPy uint64 row conflict "
          "counts. The method is synchronous and returns only compact counts; "
          "it does not materialize a dense host matrix.")
      .def(
          "__dlpack__",
          &device_commutation_matrix_dlpack,
          nb::arg("stream") = nb::none(),
          nb::arg("max_version") = nb::none(),
          nb::arg("copy") = nb::none(),
          "Export a read-only DLPack capsule for retained accelerator backends.\n\n"
          "The exported view has shape (rows, cols), dtype uint8, compact "
          "row-major layout, and a backend-specific DLPack device type. A "
          "max_version of at least (1, 0) is required because legacy DLPack "
          "capsules cannot represent the read-only flag. The returned version "
          "is the highest version supported by both producer and consumer; "
          "copy=True and stream=0 are "
          "rejected by contract. On CUDA, exporting marks the matrix so that "
          "freeing it synchronizes the device first, and the buffer is readable "
          "from the owning device only. HIP-backed matrices raise until a ROCm "
          "consumer enforces FastPauli's read-only export contract.")
      .def(
          "__dlpack_device__",
          &device_commutation_matrix_dlpack_device,
          "Return the DLPack device tuple for retained accelerator backends.")
      .def_prop_ro(
          "__cuda_array_interface__",
          [](const DeviceCommutationMatrix& matrix) {
            try {
              return device_commutation_matrix_cuda_array_interface(matrix);
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          "CUDA Array Interface view of the owned dense uint8 device buffer.\n\n"
          "On CUDA, exporting marks the matrix so that freeing it synchronizes the device "
          "first; the buffer is readable from the owning device only.");

  nb::class_<DeviceArrayView>(
      module,
      "DeviceArrayView",
      "Read-only view of one DevicePauliSum device buffer for zero-copy interop.\n\n"
      "Pass it to cupy.from_dlpack, torch.from_dlpack or cupy.asarray (PyTorch rejects "
      "read-only CUDA Array Interface input, so use torch.from_dlpack). The view and every "
      "exported array keep the owning DevicePauliSum alive. The buffer is flagged read-only, "
      "but CuPy and PyTorch do not enforce the flag: writing through a consumer array is "
      "undefined behavior. CUDA only.")
      .def_prop_ro("shape", &device_array_shape, "Array shape: (num_terms, words) or (num_terms,).")
      .def_prop_ro(
          "dtype",
          [](const DeviceArrayView& view) { return std::string(view.dtype_name); },
          "NumPy dtype name: uint64 for x and z words, complex128 for coefficients.")
      .def(
          "__dlpack__",
          &device_array_dlpack,
          nb::kw_only(),
          nb::arg("stream") = nb::none(),
          nb::arg("max_version") = nb::none(),
          nb::arg("dl_device") = nb::none(),
          nb::arg("copy") = nb::none(),
          "Export a DLPack capsule flagged read-only.\n\n"
          "max_version must be at least (1, 0), because legacy capsules cannot carry the "
          "read-only flag (CuPy 14 and current PyTorch request versioned capsules). "
          "dl_device must be None or this buffer's device; copy=True is rejected.")
      .def(
          "__dlpack_device__",
          [](const DeviceArrayView& view) { return nb::make_tuple(view.device_type, view.device); },
          "Return the DLPack device tuple.")
      .def_prop_ro(
          "__cuda_array_interface__",
          &device_array_cuda_array_interface,
          "CUDA Array Interface (version 3) of the read-only device buffer.");

  nb::class_<DevicePauliSum>(
      module,
      "DevicePauliSum",
      "Owning accelerator device mirror of a PauliSum. Construct with PauliSum.to_device().")
      .def_prop_ro(
          "num_qubits",
          &DevicePauliSum::num_qubits,
          "Number of qubits represented by this device operator.")
      .def_prop_ro(
          "num_terms",
          &DevicePauliSum::num_terms,
          "Number of terms represented by this device operator.")
      .def_prop_ro(
          "device",
          &DevicePauliSum::device,
          "Accelerator device ordinal that owns the buffers.")
      .def_prop_ro(
          "backend",
          &DevicePauliSum::backend,
          "Compiled accelerator backend that owns the buffers: 'cuda', 'hip', or 'metal'.")
      .def(
          "to_host",
          &DevicePauliSum::to_host,
          "Copy device buffers back to a new host PauliSum.")
      .def(
          "device_arrays",
          &device_pauli_sum_device_arrays,
          "Return read-only zero-copy views (x, z, coeffs) of the device buffers.\n\n"
          "x and z are (num_terms, words) uint64 packed words; coeffs is (num_terms,) "
          "complex128. Each view supports __dlpack__ and __cuda_array_interface__ and keeps "
          "this operator alive. Consumers do not enforce the read-only flag; writing through "
          "a view is undefined behavior. After an export, freeing the operator synchronizes "
          "the device first, in case a consumer still has work queued. CUDA only; HIP and "
          "Metal raise.")
      .def_static(
          "from_device_arrays",
          &device_pauli_sum_from_device_arrays,
          nb::arg("x"),
          nb::arg("z"),
          nb::arg("coeffs"),
          nb::arg("num_qubits"),
          nb::arg("device") = -1,
          "Build a DevicePauliSum by copying CUDA device arrays without a host round trip.\n\n"
          "x and z are (num_terms, words) uint64 packed words and coeffs is (num_terms,) "
          "complex128, all C-contiguous objects exposing __cuda_array_interface__ on one "
          "device (CuPy arrays or CUDA PyTorch tensors). Bits at or above num_qubits raise "
          "ValueError. device=-1 uses the arrays' device (the current device when empty). "
          "The call synchronizes the device before copying, so producers on any stream are "
          "finished, and returns after the copy completes. CUDA only.")
      .def(
          "simplify",
          [](const DevicePauliSum& op, double atol, double rtol) {
            try {
              return op.simplify(atol, rtol);
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("atol") = 1.0e-12,
          nb::arg("rtol") = 0.0,
          "Simplify a device operator and return a new DevicePauliSum on the same backend.\n\n"
          "CUDA and HIP use retained device simplify implementations. Metal source builds "
          "sort one-word operators (up to 64 qubits) of at least 12,288 terms on the GPU and "
          "sum duplicate runs in float64 on the CPU over shared memory; smaller or wider "
          "operators use the transfer-reference correctness bridge: device operator to host "
          "PauliSum, CPU PauliSum.simplify(), then a new Metal DevicePauliSum. "
          "Output follows the same canonical packed-word order and tolerance semantics as "
          "PauliSum.simplify(); both Metal paths match it bit for bit.")
      .def(
          "expectation_statevector",
          &device_expectation_statevector,
          nb::arg("psi"),
          "Compute <psi|H|psi> on the active accelerator backend.\n\n"
          "CUDA and HIP support host NumPy complex64 and complex128 statevectors. "
          "Only CUDA supports CUDA-array-interface statevector inputs. "
          "psi must be one-dimensional, contiguous, complex64 or complex128, and have length "
          "2 ** num_qubits. A CUDA-array-interface pointer and its complete byte range must "
          "belong to one runtime-recognized allocation on the operator's CUDA device.")
      .def(
          "commutes_with",
          [](const DevicePauliSum& lhs, const DevicePauliSum& rhs, nb::handle max_entries_obj) -> nb::object {
            try {
              const std::size_t max_entries = checked_size_from_python_int(
                  max_entries_obj,
                  "max_commutation_matrix_entries");
              const std::size_t entries = PauliSum::checked_commutation_matrix_entries_for_testing(
                  lhs.num_terms(),
                  rhs.num_terms(),
                  max_entries);

              if (lhs.num_terms() == 1 && rhs.num_terms() == 1) {
                std::uint8_t flag = 0;
                lhs.commutes_with_into(rhs, std::span<std::uint8_t>(&flag, 1), max_entries);
                return nb::cast(flag != 0);
              }

              nb::object array = bool_array_from_device_commutation(lhs, rhs, entries, max_entries);
              if (lhs.num_terms() == 1 || rhs.num_terms() == 1) {
                return array;
              }
              return array.attr("reshape")(nb::make_tuple(lhs.num_terms(), rhs.num_terms()));
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("other"),
          nb::arg("max_commutation_matrix_entries") = 100000000,
          "Return accelerator pairwise Pauli commutation results against another DevicePauliSum.\n\n"
          "The output shape and dense-output guardrail match PauliSum.commutes_with().")
      .def(
          "commutes_with_into",
          [](const DevicePauliSum& lhs,
             const DevicePauliSum& rhs,
             nb::handle output_obj,
             nb::handle max_entries_obj) {
            try {
              const std::size_t max_entries = checked_size_from_python_int(
                  max_entries_obj,
                  "max_commutation_matrix_entries");
              const std::size_t entries = PauliSum::checked_commutation_matrix_entries_for_testing(
                  lhs.num_terms(),
                  rhs.num_terms(),
                  max_entries);
              WritablePythonBufferView buffer(output_obj);
              lhs.commutes_with_into(
                  rhs,
                  writable_flat_bool_span(buffer.get(), entries),
                  max_entries);
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("other"),
          nb::arg("output"),
          nb::arg("max_commutation_matrix_entries") = 100000000,
          "Fill a caller-owned one-dimensional NumPy bool array with dense accelerator "
          "pairwise commutation flags.\n\n"
          "The output length must be self.num_terms * other.num_terms. Reusing this "
          "buffer avoids repeated Python allocation when callers repeatedly evaluate "
          "the same dense commutation shape.")
      .def(
          "commutes_with_device",
          [](const DevicePauliSum& lhs,
             const DevicePauliSum& rhs,
             nb::handle max_entries_obj,
             nb::object output_obj) -> nb::object {
            try {
              const std::size_t max_entries = checked_size_from_python_int(
                  max_entries_obj,
                  "max_commutation_matrix_entries");
              if (output_obj.is_none()) {
                return nb::cast(lhs.commutes_with_device(rhs, max_entries));
              }
              DeviceCommutationMatrix& output =
                  nb::cast<DeviceCommutationMatrix&>(output_obj);
              lhs.commutes_with_device_into(rhs, output, max_entries);
              return output_obj;
            } catch (const nb::cast_error&) {
              throw nb::type_error("commutes_with_device output must be a DeviceCommutationMatrix");
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("other"),
          nb::arg("max_commutation_matrix_entries") = 100000000,
          nb::arg("output") = nb::none(),
          "Return or fill a device-resident dense uint8 commutation matrix.\n\n"
          "The returned DeviceCommutationMatrix stays on the owning accelerator "
          "device until to_host() is called. Passing output reuses caller-owned FastPauli device "
          "storage with the same shape and device.")
      .def(
          "matmul",
          [](const DevicePauliSum& lhs, const DevicePauliSum& rhs, bool simplify_output, nb::handle max_terms_obj) {
            try {
              return lhs.matmul(
                  rhs,
                  simplify_output,
                  checked_size_from_python_int(max_terms_obj, "max_intermediate_terms"));
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("rhs"),
          nb::arg("simplify") = true,
          nb::arg("max_intermediate_terms") = 50000000,
          "Compose two device-resident Pauli sums with accelerator product generation.\n\n"
          "CUDA and HIP support this operation. "
          "rhs acts first and self acts second. max_intermediate_terms is enforced before "
          "allocating product buffers.");

  module.def(
      "cuda_available",
      &DevicePauliSum::cuda_available,
      "Return True when Wolfgang was built with CUDA and a CUDA device is visible.");

  module.def(
      "cuda_devices",
      []() {
        nb::list devices;
        for (const CudaDeviceInfo& device : DevicePauliSum::cuda_status().devices) {
          devices.append(cuda_device_info_to_dict(device));
        }
        return devices;
      },
      "Return visible CUDA devices as dictionaries.");



  module.def(
      "hip_available",
      &DevicePauliSum::hip_available,
      "Return True when Wolfgang was built with HIP and a HIP device is visible.");

  module.def(
      "hip_devices",
      []() {
        nb::list devices;
        for (const HipDeviceInfo& device : DevicePauliSum::hip_status().devices) {
          devices.append(hip_device_info_to_dict(device));
        }
        return devices;
      },
      "Return visible HIP devices as dictionaries.");



  module.def(
      "metal_available",
      &DevicePauliSum::metal_available,
      "Return True when Wolfgang was built with Metal and a Metal device is visible.");

  module.def(
      "metal_devices",
      []() {
        nb::list devices;
        for (const MetalDeviceInfo& device : DevicePauliSum::metal_status().devices) {
          devices.append(metal_device_info_to_dict(device));
        }
        return devices;
      },
      "Return visible Metal devices as dictionaries.");



  nb::class_<PauliSum>(
      module,
      "PauliSum",
      "Sparse sum of Pauli strings backed by packed x/z bit masks and complex128 coefficients.")
      .def(
          "__init__",
          [](PauliSum* self, long long num_qubits, long long num_terms) {
            new (self) PauliSum(
                checked_size_from_signed(num_qubits, "num_qubits"),
                checked_size_from_signed(num_terms, "num_terms"));
          },
          nb::arg("num_qubits"),
          nb::arg("num_terms") = 0,
          "Create a metadata-sized operator with zero coefficients.")
      .def_static(
          "empty",
          [](long long num_qubits) {
            return PauliSum::empty(checked_size_from_signed(num_qubits, "num_qubits"));
          },
          nb::arg("num_qubits"),
          "Create a zero-term operator with an explicit qubit count.")
      .def_static(
          "from_labels",
          [](nb::iterable labels_obj, nb::object coeffs_obj) {
            std::vector<std::string> labels = parse_labels(labels_obj);
            std::vector<std::complex<double>> coeffs = parse_coefficients(coeffs_obj, labels.size());
            try {
              return PauliSum::from_labels(labels, coeffs);
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("labels"),
          nb::arg("coeffs") = nb::none(),
          "Build from dense labels in Qiskit display order.\n\n"
          "labels must be an iterable of equal-length strings containing only I, X, Y, or Z. "
          "The right-most label character is qubit 0. coeffs may be None, one numeric scalar "
          "for a single label, or an iterable of numeric values matching labels.")
      .def_static(
          "from_sparse_list",
          [](nb::iterable triples_obj, long long num_qubits) {
            std::vector<PauliSum::SparseTerm> terms = parse_sparse_terms(triples_obj);
            try {
              return PauliSum::from_sparse_list(
                  terms,
                  checked_size_from_signed(num_qubits, "num_qubits"));
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("triples"),
          nb::arg("num_qubits"),
          "Build from sparse triples of (local_pauli_string, qubit_indices, coefficient).\n\n"
          "Each local string is aligned with its qubit_indices sequence. Indices are global qubit "
          "numbers with qubit 0 matching the right-most dense-label character. num_qubits is "
          "explicit so empty sparse input has well-defined metadata.")
      .def_static(
          "from_arrays",
          [](nb::handle x_obj, nb::handle z_obj, nb::handle coeffs_obj, long long num_qubits_obj) {
            const std::size_t num_qubits = checked_size_from_signed(num_qubits_obj, "num_qubits");
            const std::size_t words = (num_qubits + 63) / 64;
            const PackedArrayInput input = normalize_packed_arrays(x_obj, z_obj, coeffs_obj, words);
            const std::size_t num_terms = nb::cast<std::size_t>(input.coeffs.attr("shape")[0]);
            const std::size_t packed = num_terms * words;
            const auto x = contiguous_span<std::uint64_t>(input.x, packed);
            const auto z = contiguous_span<std::uint64_t>(input.z, packed);
            const auto coeffs = contiguous_span<std::complex<double>>(input.coeffs, num_terms);
            try {
              return cpu_without_gil([&] { return PauliSum::from_words(num_qubits, x, z, coeffs); });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("x"),
          nb::arg("z"),
          nb::arg("coeffs"),
          nb::arg("num_qubits"),
          "Build from packed symplectic bit masks.\n\n"
          "x and z are uint64 arrays of shape (num_terms, ceil(num_qubits / 64)); qubit q is bit "
          "q % 64 of word q // 64, so X sets x, Z sets z and Y sets both. coeffs is a 1-D numeric "
          "array of length num_terms and is converted to complex128. Bits at or above num_qubits "
          "must be zero. Data is always copied into the new operator. Other word dtypes raise "
          "TypeError rather than being cast.")
      .def_prop_ro(
          "num_qubits",
          &PauliSum::num_qubits,
          "Number of qubits represented by this operator.")
      .def_prop_ro(
          "num_terms",
          &PauliSum::num_terms,
          "Number of terms represented by this operator.")
      .def(
          "to_device",
          [](const PauliSum& op, int device, nb::object backend_obj) {
            try {
              return DevicePauliSum::from_host(
                  op,
                  parse_backend_selector(std::move(backend_obj)),
                  device);
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("device") = 0,
          nb::arg("backend").none() = nb::none(),
          "Copy this host PauliSum to accelerator device memory.\n\n"
          "backend may be None, 'auto', 'cuda', 'hip', or 'metal'. Wolfgang must be "
          "built from source with WOLFGANG_ENABLE_CUDA=ON or "
          "WOLFGANG_ENABLE_HIP=ON, or WOLFGANG_ENABLE_METAL=ON and a visible "
          "matching device must be available. "
          "CPU-only builds raise RuntimeError with rebuild guidance.")
      .def(
          "to_labels",
          [](const PauliSum& op) {
            nb::list labels;
            for (const std::string& label : op.to_labels()) {
              labels.append(nb::cast(label));
            }
            return nb::make_tuple(labels, coeff_array_from_vector(op.coeffs()));
          },
          "Export dense labels in Qiskit display order and complex128 coefficients.\n\n"
          "The returned label list preserves construction order. The coefficient array is a NumPy "
          "complex128 array owned by Python.")
      .def(
          "to_sparse_list",
          [](const PauliSum& op) {
            nb::list triples;
            for (const PauliSum::SparseTerm& term : op.to_sparse_list()) {
              nb::list indices;
              for (std::size_t qubit : term.qubit_indices) {
                indices.append(nb::cast(qubit));
              }
              triples.append(nb::make_tuple(
                  nb::cast(term.local_pauli_string),
                  indices,
                  nb::cast(term.coefficient)));
            }
            return triples;
          },
          "Export sparse triples sorted by ascending qubit index within each term.\n\n"
          "Term order is preserved. Identity terms export as an empty local string with an empty "
          "index list.")
      .def(
          "to_arrays",
          [](nb::handle_t<PauliSum> self) {
            const PauliSum& op = nb::cast<const PauliSum&>(self);
            const std::size_t terms = op.num_terms();
            const std::size_t words = op.words();
            return nb::make_tuple(
                readonly_view(self, op.x_words().data(), {terms, words}),
                readonly_view(self, op.z_words().data(), {terms, words}),
                readonly_view(self, op.coeffs().data(), {terms}));
          },
          "Return (x, z, coeffs) as read-only NumPy views of the packed storage.\n\n"
          "x and z have dtype uint64 and shape (num_terms, ceil(num_qubits / 64)); coeffs is "
          "complex128 with shape (num_terms,). The views share memory with this immutable "
          "operator and keep it alive; call .copy() for writable arrays. "
          "PauliSum.from_arrays(*op.to_arrays(), op.num_qubits) reproduces op.")
      .def(
          "simplify",
          [](const PauliSum& op, double atol, double rtol) {
            wolfgang::ensure_cpu_backend_supports_simplify();
            try {
              return cpu_without_gil([&] { return op.simplify(atol, rtol); });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("atol") = 1.0e-12,
          nb::arg("rtol") = 0.0,
          "Combine duplicate Pauli strings and return canonical packed-word order.\n\n"
          "Terms with abs(coefficient) <= atol + rtol * max_abs_input_coefficient are dropped. "
          "Negative or non-finite tolerances and non-finite coefficient components raise ValueError. "
          "Duplicate accumulation overflow raises OverflowError. Idempotence requires rtol=0.")
      .def(
          "__add__",
          [](const PauliSum& lhs, const PauliSum& rhs) {
            ensure_scalar_cpu_operation("__add__");
            try {
              return cpu_without_gil([&] { return lhs.add(rhs); });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::is_operator(),
          "Concatenate two PauliSum objects without implicit simplify.")
      .def(
          "__sub__",
          [](const PauliSum& lhs, const PauliSum& rhs) {
            ensure_scalar_cpu_operation("__sub__");
            try {
              return cpu_without_gil([&] { return lhs.add(rhs.negate()); });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::is_operator(),
          "Concatenate self with the negated terms of rhs, without implicit simplify.")
      .def(
          "__neg__",
          [](const PauliSum& op) {
            ensure_scalar_cpu_operation("__neg__");
            return cpu_without_gil([&] { return op.negate(); });
          },
          "Return the operator with every coefficient's sign flipped exactly.")
      .def(
          "__len__",
          &PauliSum::num_terms,
          "Number of terms; equal to num_terms.")
      .def(
          "__getitem__",
          [](const PauliSum& op, nb::handle key) {
            std::vector<std::size_t> indices;
            const auto num_terms = static_cast<Py_ssize_t>(op.num_terms());
            if (PySlice_Check(key.ptr())) {
              Py_ssize_t start = 0;
              Py_ssize_t stop = 0;
              Py_ssize_t step = 0;
              if (PySlice_Unpack(key.ptr(), &start, &stop, &step) != 0) {
                throw nb::python_error();
              }
              const Py_ssize_t count = PySlice_AdjustIndices(num_terms, &start, &stop, step);
              indices.reserve(static_cast<std::size_t>(count));
              for (Py_ssize_t i = 0; i < count; ++i) {
                indices.push_back(static_cast<std::size_t>(start + i * step));
              }
            } else {
              PyObject* index_obj = PyNumber_Index(key.ptr());
              if (index_obj == nullptr) {
                PyErr_Clear();
                throw nb::type_error("PauliSum indices must be integers or slices");
              }
              Py_ssize_t index = PyNumber_AsSsize_t(index_obj, PyExc_IndexError);
              Py_DECREF(index_obj);
              if (index == -1 && PyErr_Occurred()) {
                throw nb::python_error();
              }
              if (index < 0) {
                index += num_terms;
              }
              if (index < 0 || index >= num_terms) {
                throw nb::index_error("PauliSum term index out of range");
              }
              indices.push_back(static_cast<std::size_t>(index));
            }
            return op.select_terms(indices);
          },
          nb::arg("key").none(),
          "Return the term at an integer index, or the terms in a slice, as a new PauliSum.\n\n"
          "Negative indices count from the end. Out-of-range integers raise IndexError; "
          "iteration yields one single-term PauliSum per term.")
      .def(
          "__eq__",
          [](const PauliSum& lhs, nb::handle rhs) -> nb::object {
            if (!nb::isinstance<PauliSum>(rhs)) {
              return nb::borrow(Py_NotImplemented);
            }
            return nb::cast(lhs.structurally_equal(nb::cast<const PauliSum&>(rhs)));
          },
          nb::is_operator(),
          "Structural equality: same num_qubits and the same terms in the same order with equal "
          "coefficients. Use equiv() to compare operators mathematically.")
      .def(
          "equiv",
          [](const PauliSum& lhs, const PauliSum& rhs, double atol, double rtol) {
            ensure_scalar_cpu_operation("equiv");
            if (lhs.num_qubits() != rhs.num_qubits()) {
              return false;
            }
            try {
              return cpu_without_gil([&] {
                return lhs.add(rhs.negate()).simplify(atol, rtol).num_terms() == 0;
              });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("other"),
          nb::arg("atol") = 1.0e-12,
          nb::arg("rtol") = 0.0,
          "Return True when self and other are the same operator up to tolerance.\n\n"
          "Equivalent to (self - other).simplify(atol, rtol).num_terms == 0, so term order and "
          "unmerged duplicates do not matter. Operators on different qubit counts are never "
          "equivalent.")
      .def(
          "adjoint",
          [](const PauliSum& op) {
            ensure_scalar_cpu_operation("adjoint");
            return cpu_without_gil([&] { return op.adjoint(); });
          },
          "Return the Hermitian adjoint: coefficients are conjugated, terms and order unchanged.")
      .def(
          "__mul__",
          [](const PauliSum& op, nb::handle scalar_obj) {
            ensure_scalar_cpu_operation("__mul__");
            const auto scalar = parse_complex_value(scalar_obj, "scalar");
            return cpu_without_gil([&] { return op.scalar_multiply(scalar); });
          },
          nb::is_operator(),
          "Scale coefficients by a Python numeric scalar.")
      .def(
          "__rmul__",
          [](const PauliSum& op, nb::handle scalar_obj) {
            ensure_scalar_cpu_operation("__rmul__");
            const auto scalar = parse_complex_value(scalar_obj, "scalar");
            return cpu_without_gil([&] { return op.scalar_multiply(scalar); });
          },
          nb::is_operator(),
          "Scale coefficients by a Python numeric scalar.")
      .def(
          "matmul",
          [](const PauliSum& lhs, const PauliSum& rhs, bool simplify_output, nb::handle max_terms_obj) {
            ensure_scalar_cpu_operation("matmul");
            try {
              const auto limit = checked_size_from_python_int(max_terms_obj, "max_intermediate_terms");
              return cpu_without_gil([&] { return lhs.matmul(rhs, simplify_output, limit); });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("rhs"),
          nb::arg("simplify") = true,
          nb::arg("max_intermediate_terms") = 50000000,
          "Compose two PauliSum objects with matrix multiplication semantics.\n\n"
          "rhs acts first and self acts second. Set simplify=False to preserve nested-loop "
          "product order before duplicate reduction.")
      .def(
          "__matmul__",
          [](const PauliSum& lhs, const PauliSum& rhs) {
            ensure_scalar_cpu_operation("__matmul__");
            try {
              return cpu_without_gil([&] { return lhs.matmul(rhs); });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::is_operator(),
          "Compose two PauliSum objects with matrix multiplication semantics.")
      .def(
          "commutator",
          [](const PauliSum& lhs, const PauliSum& rhs, bool simplify_output, nb::handle max_terms_obj) {
            ensure_scalar_cpu_operation("commutator");
            try {
              const auto limit = checked_size_from_python_int(max_terms_obj, "max_intermediate_terms");
              return cpu_without_gil([&] { return lhs.commutator(rhs, simplify_output, limit); });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("other"),
          nb::arg("simplify") = true,
          nb::arg("max_intermediate_terms") = 50000000,
          "Return the commutator [self, other] = self @ other - other @ self.\n\n"
          "Commuting term pairs cancel exactly and are never generated, so this is faster and "
          "more precise than composing two products. Each anticommuting pair contributes "
          "2 * a * b * (P @ Q). With simplify=False the surviving products keep self-outer, "
          "other-inner order. max_intermediate_terms bounds len(self) * len(other). Non-finite "
          "coefficients raise ValueError.")
      .def(
          "anticommutator",
          [](const PauliSum& lhs, const PauliSum& rhs, bool simplify_output, nb::handle max_terms_obj) {
            ensure_scalar_cpu_operation("anticommutator");
            try {
              const auto limit = checked_size_from_python_int(max_terms_obj, "max_intermediate_terms");
              return cpu_without_gil([&] { return lhs.anticommutator(rhs, simplify_output, limit); });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("other"),
          nb::arg("simplify") = true,
          nb::arg("max_intermediate_terms") = 50000000,
          "Return the anticommutator {self, other} = self @ other + other @ self.\n\n"
          "Only commuting term pairs contribute, each as 2 * a * b * (P @ Q).")
      .def(
          "commutes_with",
          [](const PauliSum& lhs, const PauliSum& rhs, nb::handle max_entries_obj) -> nb::object {
            ensure_supported_cpu_backend();
            try {
              const auto limit = checked_size_from_python_int(max_entries_obj, "max_commutation_matrix_entries");
              const auto flags = cpu_without_gil([&] { return lhs.commutes_with(rhs, limit); });

              if (lhs.num_terms() == 1 && rhs.num_terms() == 1) {
                return nb::cast(!flags.empty() && flags.front() != 0);
              }

              nb::object array = bool_array_from_vector(flags);
              if (lhs.num_terms() == 1 || rhs.num_terms() == 1) {
                return array;
              }
              return array.attr("reshape")(nb::make_tuple(lhs.num_terms(), rhs.num_terms()));
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("other"),
          nb::arg("max_commutation_matrix_entries") = 100000000,
          "Return Pauli commutation results against another PauliSum.\n\n"
          "The result is bool for single-term inputs, a NumPy bool vector when exactly one "
          "operand has one term, and a NumPy bool matrix for many-to-many inputs. Large dense "
          "matrix requests raise ValueError before allocating output.")
      .def(
          "group_commuting",
          [](const PauliSum& op, std::string mode, std::string strategy, nb::handle max_terms_obj) {
            if (mode == "qwc") {
              ensure_scalar_cpu_operation("group_commuting(mode='qwc')");
            } else {
              ensure_supported_cpu_backend();
            }
            try {
              const auto limit = checked_size_from_python_int(max_terms_obj, "max_terms_for_graph");
              const auto native_groups = cpu_without_gil([&] { return op.group_commuting(mode, strategy, limit); });
              nb::list groups;
              for (const PauliSum& group : native_groups) {
                groups.append(nb::cast(group));
              }
              return groups;
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("mode") = "qwc",
          nb::arg("strategy") = "largest_first",
          nb::arg("max_terms_for_graph") = 50000,
          "Greedily partition terms into deterministic commuting groups.\n\n"
          "mode='qwc' requires qubit-wise compatibility within each group. mode='full' requires "
          "global Pauli commutation within each group. strategy currently supports "
          "'largest_first'.")
      .def(
          "expectation_statevector",
          &expectation_statevector,
          nb::arg("psi"),
          "Compute <psi|H|psi> for a NumPy statevector or a batch of statevectors.\n\n"
          "psi must be C-contiguous with dtype complex64 or complex128. A 1-D psi of length "
          "2 ** num_qubits returns a Python complex; a 2-D psi of shape (batch, 2 ** num_qubits) "
          "returns a complex128 array with one value per row. Requires num_qubits <= 63.")
      .def(
          "variance_statevector",
          &variance_statevector,
          nb::arg("psi"),
          "Return the variance of H in the normalized state psi / ||psi||, clamped at zero.\n\n"
          "Computes <H^dagger H> / N - |<H>|^2 / N^2 with N = <psi|psi>; a zero state raises "
          "ValueError. "
          "Accepts the same 1-D or batched inputs as expectation_statevector and returns a float "
          "or a float64 array.")
      .def(
          "apply_statevector",
          &apply_statevector,
          nb::arg("psi"),
          "Return H @ psi as a new complex128 array of the same shape.\n\n"
          "Basis index bit q is qubit q, matching dense labels whose right-most character is "
          "qubit 0. Accepts one statevector or a 2-D batch; requires num_qubits <= 63.")
      .def(
          "to_matrix",
          [](const PauliSum& op, bool sparse, nb::handle max_entries_obj) -> nb::object {
            ensure_scalar_cpu_operation("to_matrix");
            const auto limit = checked_size_from_python_int(max_entries_obj, "max_entries");
            try {
              if (!sparse) {
                const std::size_t entries = op.dense_matrix_entries(limit);
                const std::size_t dimension = std::size_t{1} << op.num_qubits();
                auto [array, out] =
                    new_numpy_array<std::complex<double>>({dimension, dimension}, "complex128");
                const std::span<std::complex<double>> out_span(out, entries);
                cpu_without_gil([&] { op.to_dense_matrix_into(out_span); });
                return array;
              }
              nb::object scipy_sparse;
              try {
                scipy_sparse = nb::module_::import_("scipy.sparse");
              } catch (nb::python_error& error) {
                nb::raise_from(
                    error,
                    PyExc_ImportError,
                    "to_matrix(sparse=True) requires SciPy; install it with "
                    "`python -m pip install 'wolfgang-quantum[scipy]'`");
              }
              const auto csr = cpu_without_gil([&] { return op.to_csr_matrix(limit); });
              nb::object data = coeff_array_from_vector(csr.data);
              nb::object indices = numeric_array_from_vector(csr.indices, "int64");
              nb::object indptr = numeric_array_from_vector(csr.indptr, "int64");
              return scipy_sparse.attr("csr_matrix")(
                  nb::make_tuple(data, indices, indptr),
                  nb::arg("shape") = nb::make_tuple(csr.dimension, csr.dimension));
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("sparse") = false,
          nb::arg("max_entries") = 16777216,
          "Return the 2 ** n x 2 ** n matrix with entry [r, c] = <r|H|c>.\n\n"
          "sparse=False returns a dense complex128 NumPy array and raises ValueError before "
          "allocating when 4 ** num_qubits > max_entries. sparse=True returns a SciPy CSR matrix "
          "with sorted column indices and needs SciPy; it raises ValueError when "
          "2 ** num_qubits * (distinct X patterns) > max_entries. Entries that cancel to exactly "
          "zero are omitted from the sparse form.")
      .def(
          "estimate_counts",
          [](const PauliSum& op, nb::handle counts_obj, nb::object basis_obj) {
            ensure_scalar_cpu_operation("estimate_counts");
            std::vector<std::string> bitstrings;
            std::vector<double> counts;
            parse_z_counts_mapping(counts_obj, bitstrings, counts);
            std::string basis;
            if (!basis_obj.is_none()) {
              try {
                basis = nb::cast<std::string>(basis_obj);
              } catch (const nb::cast_error&) {
                throw nb::type_error("basis must be a string or None");
              }
            }
            try {
              const auto estimate =
                  cpu_without_gil([&] { return op.estimate_counts(bitstrings, counts, basis); });
              return nb::make_tuple(estimate.mean, estimate.standard_error);
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("counts"),
          nb::arg("basis").none() = nb::none(),
          "Estimate <H> and its standard error from shot counts in a local measurement basis.\n\n"
          "basis is a dense-label-order string of X, Y, Z or I giving the basis each qubit was "
          "measured in (the right-most character is qubit 0); None means all Z. Outcome bit 0 is "
          "the +1 eigenstate of that qubit's basis Pauli (h rotates X, sdg then h rotates Y). "
          "Every term must act "
          "on each qubit as I or as that qubit's basis letter. counts maps measured bitstrings to "
          "raw shot counts. Returns (mean, standard_error), where standard_error is the unbiased "
          "sample standard deviation of the per-shot estimator divided by sqrt(total shots), or "
          "NaN for a single shot.")
      .def(
          "expectation_z_counts",
          [](const PauliSum& op, nb::handle counts_obj) {
            ensure_scalar_cpu_operation("expectation_z_counts");
            std::vector<std::string> bitstrings;
            std::vector<double> counts;
            parse_z_counts_mapping(counts_obj, bitstrings, counts);
            try {
              return cpu_without_gil([&] { return op.expectation_z_counts(bitstrings, counts); });
            } catch (const std::invalid_argument& error) {
              translate_invalid_argument(error);
            }
          },
          nb::arg("counts"),
          "Compute expectation over computational-basis counts.\n\n"
          "counts must map dense bitstrings to finite non-negative numeric counts. Dense "
          "bitstrings follow the label convention: the right-most bit is qubit 0. Only "
          "diagonal Pauli terms are supported by this initial CPU path.");

  // Structural __eq__ makes PauliSum unhashable, matching Python's rule for
  // classes that define equality without a consistent hash.
  module.attr("PauliSum").attr("__hash__") = nb::none();
}

}  // namespace wolfgang::python
