#pragma once
#include <cstddef>
#include <string>
#include <vector>

// Fixed-capacity LRU of compiled NVRTC modules, one pool per analysis type.
// T must expose `std::string key` plus a `module` handle testable against null.
// Unloading is left to the caller (evicted entries are handed back) so that this
// header stays free of CUDA headers and can be tested on its own.
template <class T>
class ModuleLru {
public:
    explicit ModuleLru(std::size_t capacity) : cap_(capacity ? capacity : 1) {}

    // Cache hit: copies the entry out and promotes it to most-recent.
    bool take(const std::string& key, T& out) {
        for (std::size_t i = 0; i < items_.size(); ++i) {
            if (items_[i].key != key) continue;
            if (i + 1 != items_.size()) {
                T hit = items_[i];
                items_.erase(items_.begin() + i);
                items_.push_back(hit);
            }
            out = items_.back();
            return true;
        }
        return false;
    }

    // Inserts as most-recent; whatever no longer fits lands in `evicted` for the
    // caller to unload. `pinned` is never evicted -- a run may still hold its kernels.
    void insert(const T& entry, const std::string& pinned, std::vector<T>& evicted) {
        for (std::size_t i = 0; i < items_.size(); ++i) {
            if (items_[i].key != entry.key) continue;
            evicted.push_back(items_[i]);   // same key compiled twice: the loser must still be unloaded
            items_.erase(items_.begin() + i);
            break;
        }
        items_.push_back(entry);
        while (items_.size() > cap_) {
            std::size_t victim = items_.size();
            for (std::size_t i = 0; i + 1 < items_.size(); ++i) {   // never the entry just inserted
                if (items_[i].key == pinned) continue;
                victim = i;
                break;
            }
            if (victim == items_.size()) break;   // all pinned: overshoot rather than unload live code
            evicted.push_back(items_[victim]);
            items_.erase(items_.begin() + victim);
        }
    }

    // Hands every entry over for unloading and empties the pool.
    std::vector<T> drain() {
        std::vector<T> all;
        all.swap(items_);
        return all;
    }

    std::size_t size() const { return items_.size(); }
    std::size_t capacity() const { return cap_; }

private:
    std::vector<T> items_;   // least-recent first, most-recent last
    std::size_t    cap_;
};
