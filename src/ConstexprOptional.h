#pragma once

#include <type_traits>
#include <utility>

// Class to simulate std::optional with compile-time evaluation and prevent
// accidental access when no value is set
template <bool Cond, typename T>
class ConstexprOptional {
  // No set value placeholder
  struct EmptyState {};

  // Underlying data
  using StorageType = std::conditional_t<Cond, T, EmptyState>;
  StorageType object_;

 public:
  // Disabled / default ctor
  template <bool C = Cond, typename = std::enable_if_t<!C>>
  constexpr ConstexprOptional() : object_{} {}

  // Static empty() factory only enabled when cond == false
  template <bool C = Cond, typename = std::enable_if_t<!C>>
  static constexpr ConstexprOptional empty() {
    return ConstexprOptional{};
  }

  // Main ctor for enabled state
  template <bool C = Cond, typename = std::enable_if_t<C>, typename... Args>
  explicit constexpr ConstexprOptional(Args&&... args)
      : object_{std::forward<Args>(args)...} {}

  // Dereference operators (only enabled when cond == true)
  template <bool C = Cond, typename = std::enable_if_t<C>>
  constexpr const T& operator*() const noexcept {
    return object_;
  }

  template <bool C = Cond, typename = std::enable_if_t<C>>
  constexpr T& operator*() noexcept {
    return object_;
  }

  // Removing these to avoid dereferencing overhead
  // template <bool C = Cond, typename = std::enable_if_t<C>>
  // constexpr const T* operator->() const noexcept {
  //   return &object_;
  // }
  //
  // template <bool C = Cond, typename = std::enable_if_t<C>>
  // constexpr T* operator->() noexcept {
  //   return &object_;
  // }
};
