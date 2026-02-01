#ifndef CV_LM_CONSTEXPROPTIONAL_H
#define CV_LM_CONSTEXPROPTIONAL_H

#include <RcppArmadillo.h>

#include <type_traits>
#include <utility>

// Class to simulate std::optional with compile-time evaluation and prevent
// accidental access when no value is set
template <bool Enabled, typename T>
class ConstexprOptional {
 public:
  // Retrieve whether the object is enabled or not
  static constexpr bool isEnabled{Enabled};

 private:
  // No set value placeholder
  struct EmptyState {};

  // Underlying data
  std::conditional_t<Enabled, T, EmptyState> object_;

  // Detect if the type is an arma type
  template <typename U>
  static constexpr bool is_arma_type_v{arma::is_arma_type<U>::value};

 public:
  // Disabled-state ctor
  template <bool C = Enabled, std::enable_if_t<!C, int> = 0, typename... Args>
  explicit constexpr ConstexprOptional(Args&&...) : object_{} {}

  // Enabled-state ctor for non-armadillo types
  template <bool C = Enabled,
            std::enable_if_t<C && !is_arma_type_v<T>, int> = 0,
            typename... Args>
  explicit constexpr ConstexprOptional(Args&&... args)
      : object_{std::forward<Args>(args)...} {}

  // Enabled-state ctor for armadillo types (the compiler prefers the
  // initializer list ctor when using {} when we actually want to pre-allocate
  template <bool C = Enabled, std::enable_if_t<C && is_arma_type_v<T>, int> = 0,
            typename... Args>
  explicit constexpr ConstexprOptional(Args&&... args)
      : object_(std::forward<Args>(args)...) {}

  // Retrieve underlying data (only enabled when cond == true)
  template <bool C = Enabled, typename = std::enable_if_t<C>>
  [[nodiscard]] constexpr const T& value() const noexcept {
    static_assert(
        Enabled && C,
        "Attempting to retrieve an unset value of a ConstexprOptional object.");
    return object_;
  }

  template <bool C = Enabled, typename = std::enable_if_t<C>>
  [[nodiscard]] constexpr T& value() noexcept {
    static_assert(
        Enabled && C,
        "Attempting to retrieve an unset value of a ConstexprOptional object.");
    return object_;
  }
};

#endif  // CV_LM_CONSTEXPROPTIONAL_H
