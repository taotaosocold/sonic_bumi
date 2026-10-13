#ifndef NOETIX_COMMON_REALTIME_BUFFER_HPP_
#define NOETIX_COMMON_REALTIME_BUFFER_HPP_

#include <array>
#include <atomic>
#include <cstddef>

namespace noetix {

/**
 * @brief Allocation-free SPSC latest-value buffer after initialize().
 *
 * One writer and one reader exchange snapshots through three preallocated
 * slots. The reader publishes the slot it is currently copying, so the writer
 * never overwrites memory being read. The writer never waits for the reader.
 *
 * initialize() must be called before the real-time loop starts. T may contain
 * std::vector members, but their capacities should already be sized by the
 * initial value. With fixed-size control data this avoids new/delete in the
 * periodic loop.
 *
 * @note Exactly one logical writer and one logical reader are supported.
 */
template <class T>
class RealtimeBuffer {
 public:
  RealtimeBuffer() = default;

  explicit RealtimeBuffer(const T& initial) { initialize(initial); }

  void initialize(const T& initial) {
    for (auto& slot : slots_) {
      slot = initial;
    }
    reading_.store(kNoSlot, std::memory_order_relaxed);
    published_.store(0, std::memory_order_release);
    initialized_.store(true, std::memory_order_release);
  }

  bool initialized() const noexcept {
    return initialized_.load(std::memory_order_acquire);
  }

  /**
   * @brief Publish a new snapshot without blocking the reader.
   * @return false only when initialize() has not been called.
   */
  bool write(const T& value) {
    if (!initialized()) {
      return false;
    }

    const int current = published_.load(std::memory_order_acquire);
    const int reading = reading_.load(std::memory_order_acquire);

    int target = kNoSlot;
    for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
      if (i != current && i != reading) {
        target = i;
        break;
      }
    }

    // With three slots and a single reader there is always one slot that is
    // neither published nor being copied.
    if (target == kNoSlot) {
      return false;
    }

    slots_[static_cast<std::size_t>(target)] = value;
    published_.store(target, std::memory_order_release);
    return true;
  }

  /**
   * @brief Copy the latest complete snapshot into caller-owned storage.
   *
   * The retry only covers the tiny window between observing the published
   * slot and claiming it for reading. Once claimed, the writer excludes that
   * slot until the copy completes.
   */
  bool read(T& out) const {
    if (!initialized()) {
      return false;
    }

    for (;;) {
      const int index = published_.load(std::memory_order_acquire);
      if (index == kNoSlot) {
        return false;
      }

      reading_.store(index, std::memory_order_release);

      if (published_.load(std::memory_order_acquire) != index) {
        reading_.store(kNoSlot, std::memory_order_release);
        continue;
      }

      out = slots_[static_cast<std::size_t>(index)];
      reading_.store(kNoSlot, std::memory_order_release);
      return true;
    }
  }

 private:
  static constexpr int kNoSlot = -1;

  std::array<T, 3> slots_{};
  std::atomic<int> published_{kNoSlot};
  mutable std::atomic<int> reading_{kNoSlot};
  std::atomic<bool> initialized_{false};
};

}  // namespace noetix

#endif  // NOETIX_COMMON_REALTIME_BUFFER_HPP_
