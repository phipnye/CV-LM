#ifndef CV_LM_CONSTEXPROPTIONAL_H
#define CV_LM_CONSTEXPROPTIONAL_H

#include <RcppEigen.h>

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

  // Disabled / default ctor
  template <bool C = Cond, typename = std::enable_if_t<!C>>
  constexpr ConstexprOptional() : object_{} {}

  // Main ctor for enabled state
  template <bool C = Cond, typename = std::enable_if_t<C>, typename... Args>
  explicit constexpr ConstexprOptional(Args&&... args)
      : object_{std::forward<Args>(args)...} {}

 public:
  // Empty state creator
  template <bool C = Cond, typename = std::enable_if_t<!C>>
  static constexpr ConstexprOptional empty() {
    return ConstexprOptional{};
  }

  // Public facing constuctor dispatch
  template <typename... Args>
  static constexpr ConstexprOptional make(Args&&... args) {
    if constexpr (Cond) {
      return ConstexprOptional{std::forward<Args>(args)...};  // enabled state
    } else {
      return empty();  // empty/disabled state
    }
  }

  // Helper to clone data buffers by copying their dimensions without copying
  // any data (copies the data for non-Eigen Matrix or Vector types)
  [[nodiscard]] constexpr ConstexprOptional clone() const {
    if constexpr (Cond) {
      if constexpr (std::is_base_of_v<
                        Eigen::PlainObjectBase<std::remove_cv_t<T>>,
                        std::remove_cv_t<T>>) {
        return make(object_.rows(), object_.cols());  // works for vectors too
      } else {
        return make(object_);
      }
    } else {
      return empty();
    }
  }

  // Retrieve underlying data (only enabled when cond == true)
  template <bool C = Cond, typename = std::enable_if_t<C>>
  [[nodiscard]] constexpr const T& value() const noexcept {
    static_assert(
        Cond && C,
        "Attempting to retrieve an unset value of a ConstexprOptional object.");
    return object_;
  }

  template <bool C = Cond, typename = std::enable_if_t<C>>
  [[nodiscard]] constexpr T& value() noexcept {
    static_assert(
        Cond && C,
        "Attempting to retrieve an unset value of a ConstexprOptional object.");
    return object_;
  }
};

#endif  // CV_LM_CONSTEXPROPTIONAL_H
