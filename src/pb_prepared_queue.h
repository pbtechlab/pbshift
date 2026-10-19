#pragma once
#include <cstddef>
#include <vector>
namespace pbshift {
template <class T>
class PreparedQueue {
    std::vector<T> storage_;
    std::size_t head_ = 0, count_ = 0;

   public:
    void reserve(std::size_t requested) {
        if (requested <= storage_.size()) return;
        std::size_t capacity = 8;
        while (capacity < requested) capacity *= 2;
        std::vector<T> next(capacity);
        for (std::size_t i = 0; i < count_; ++i) next[i] = (*this)[i];
        storage_.swap(next);
        head_ = 0;
    }
    void push_back(const T& value) {
        if (count_ == storage_.size()) reserve(storage_.empty() ? 8 : storage_.size() * 2);
        storage_[(head_ + count_) & (storage_.size() - 1)] = value;
        ++count_;
    }
    void pop_front() {
        head_ = (head_ + 1) & (storage_.size() - 1);
        --count_;
    }
    void clear() { head_ = count_ = 0; }
    bool empty() const { return count_ == 0; }
    std::size_t size() const { return count_; }
    T& operator[](std::size_t i) { return storage_[(head_ + i) & (storage_.size() - 1)]; }
    const T& operator[](std::size_t i) const { return storage_[(head_ + i) & (storage_.size() - 1)]; }
    T& front() { return (*this)[0]; }
    const T& front() const { return (*this)[0]; }
    struct Iterator {
        const PreparedQueue* queue;
        std::size_t index;
        const T& operator*() const { return (*queue)[index]; }
        Iterator& operator++() {
            ++index;
            return *this;
        }
        bool operator!=(const Iterator& other) const { return index != other.index; }
    };
    Iterator begin() const { return {this, 0}; }
    Iterator end() const { return {this, count_}; }
};
}  // namespace pbshift
