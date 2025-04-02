#include <algorithm>
#include <cctype>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

template <typename Key, typename Value>
struct KV {
    Key key;
    Value value;
    auto operator<=>(const KV&) const = default;
};

template <typename Key, typename Value>
bool operator==(const KV<Key, Value>& lhs, const KV<Key, Value>& rhs) {
    return lhs.key == rhs.key;
}

namespace std {
template <typename Key, typename Value>
struct hash<KV<Key, Value>> {
    std::size_t operator()(const KV<Key, Value>& kv) const noexcept {
        return std::hash<Key>{}(kv.key);
    }
};
}  // namespace std

template <typename T>
class DataFlow {
   public:
    using value_type = T;
    using Consumer = std::function<void(const T&)>;
    std::function<void(Consumer)> for_each;
    mutable std::optional<std::vector<T>> cache;

    explicit DataFlow(std::function<void(Consumer)> f)
        : for_each(std::move(f)) {}

    template <typename F>
    auto operator|(F f) const {
        return f(*this);
    }

    std::vector<T> to_vector() const {
        fill_cache_if_needed();
        return *cache;
    }

    auto begin() const {
        fill_cache_if_needed();
        return cache->begin();
    }
    auto end() const { return cache->end(); }

   private:
    void fill_cache_if_needed() const {
        if (!cache) {
            std::vector<T> vec;
            for_each([&vec](const T& item) { vec.push_back(item); });
            cache = std::move(vec);
        }
    }
};

template <typename T, typename F>
auto operator|(const DataFlow<T>& flow, F f) {
    return f(flow);
}

inline DataFlow<std::filesystem::path> Dir(const std::string& path,
                                           bool recursive) {
    return DataFlow<std::filesystem::path>(
        [path, recursive](
            std::function<void(const std::filesystem::path&)> consumer) {
            if (recursive) {
                for (auto& entry :
                     std::filesystem::recursive_directory_iterator(path)) {
                    consumer(entry.path());
                }
            } else {
                for (auto& entry : std::filesystem::directory_iterator(path)) {
                    consumer(entry.path());
                }
            }
        });
}

template <typename Predicate>
struct FilterWrapper {
    Predicate pred;
    explicit FilterWrapper(Predicate p) : pred(p) {}

    template <typename T>
    DataFlow<T> operator()(const DataFlow<T>& input) const {
        return DataFlow<T>(
            [this, &input](std::function<void(const T&)> consumer) {
                input.for_each([this, &consumer](const T& item) {
                    if (pred(item)) consumer(item);
                });
            });
    }
};

template <typename Predicate>
auto Filter(Predicate pred) {
    return FilterWrapper<Predicate>(pred);
}

template <typename TransformFunc>
struct TransformWrapper {
    TransformFunc func;
    explicit TransformWrapper(TransformFunc f) : func(f) {}

    template <typename T,
              typename U = std::invoke_result_t<TransformFunc, const T&>>
    DataFlow<U> operator()(const DataFlow<T>& input) const {
        auto func_copy = func;
        return DataFlow<U>(
            [input, func_copy](std::function<void(const U&)> consumer) {
                input.for_each([func_copy, &consumer](const T& item) {
                    U result = func_copy(item);
                    consumer(result);
                });
            });
    }
};

template <typename TransformFunc>
auto Transform(TransformFunc f) {
    return TransformWrapper<TransformFunc>(f);
}

struct OpenFilesWrapper {
    template <typename T = std::filesystem::path>
    DataFlow<std::string> operator()(const DataFlow<T>& input) const {
        return DataFlow<std::string>(
            [&input](std::function<void(const std::string&)> consumer) {
                input.for_each([&consumer](const T& path) {
                    std::ifstream file(path);
                    if (!file.is_open()) return;
                    std::string line;
                    while (std::getline(file, line)) consumer(line);
                });
            });
    }
};

inline auto OpenFiles() { return OpenFilesWrapper(); }

struct SplitWrapper {
    std::string delimiters;
    explicit SplitWrapper(const std::string& delims) : delimiters(delims) {}

    DataFlow<std::string> operator()(const DataFlow<std::string>& input) const {
        return DataFlow<std::string>(
            [this, input](std::function<void(const std::string&)> consumer) {
                input.for_each([this, consumer](const std::string& line) {
                    std::string token;
                    for (char c : line) {
                        if (delimiters.find(c) != std::string::npos) {
                            consumer(token);
                            token.clear();
                        } else {
                            token.push_back(c);
                        }
                    }
                    if (!token.empty()) consumer(token);
                });
            });
    }

    template <typename U,
              typename = std::enable_if_t<!std::is_same_v<U, std::string>>>
    DataFlow<std::string> operator()(const DataFlow<U>& input) const {
        return DataFlow<std::string>(
            [this, &input](std::function<void(const std::string&)> consumer) {
                input.for_each([this, &consumer](const U& item) {
                    std::string line;
                    if constexpr (std::is_same_v<U, std::stringstream>)
                        line = item.str();
                    else
                        line = static_cast<std::string>(item);
                    std::string token;
                    for (char c : line) {
                        if (delimiters.find(c) != std::string::npos) {
                            if (!token.empty()) {
                                consumer(token);
                                token.clear();
                            }
                        } else {
                            token.push_back(c);
                        }
                    }
                    if (!token.empty()) consumer(token);
                });
            });
    }
};

inline auto Split(const std::string& delimiters) {
    return SplitWrapper(delimiters);
}

struct OutWrapper {
    std::ostream& os;
    std::string separator;
    OutWrapper(std::ostream& out, const std::string& sep)
        : os(out), separator(sep) {}

    template <typename T>
    DataFlow<T> operator()(const DataFlow<T>& input) const {
        bool first = true;
        input.for_each([this, &first](const T& item) {
            if (!first) os << separator;
            os << item;
            first = false;
        });
        os << separator;
        return input;
    }
};

inline auto Out(std::ostream& os, const std::string& separator) {
    return OutWrapper(os, separator);
}
inline auto Out(std::ostream& os, char separator) {
    return Out(os, std::string(1, separator));
}
inline auto Out(std::ostream& os) { return Out(os, "\n"); }

struct WriteWrapper {
    std::ofstream ofs;
    std::string separator;
    WriteWrapper(const std::string& filename, const std::string& sep)
        : ofs(filename), separator(sep) {}

    template <typename T>
    DataFlow<T> operator()(DataFlow<T>& input) {
        bool first = true;
        input.for_each([this, &first](const T& item) {
            if (!first) ofs << separator;
            ofs << item;
            first = false;
        });
        ofs << separator;
        return input;
    }
};

inline auto Write(const std::string& filename, const std::string& separator) {
    return WriteWrapper(filename, separator);
}
inline auto Write(const std::string& filename, char separator) {
    return WriteWrapper(filename, std::string(1, separator));
}
inline auto Write(std::ostream& os, const std::string& separator) {
    return Out(os, separator);
}
inline auto Write(std::ostream& os, char separator) {
    return Out(os, separator);
}

template <typename Container>
auto AsDataFlow(Container& container) -> std::enable_if_t<
    !std::is_base_of_v<
        std::istream,

        std::conditional_t<
            std::is_pointer_v<typename Container::value_type>,
            std::remove_pointer_t<typename Container::value_type>,
            typename Container::value_type>>,
    DataFlow<typename Container::value_type>> {
    using T = typename Container::value_type;
    return DataFlow<T>([&container](std::function<void(const T&)> consumer) {
        for (auto& item : container) {
            consumer(item);
        }
    });
}

template <typename Container>
auto AsDataFlow(Container& container) -> std::enable_if_t<
    std::is_base_of_v<std::iostream,
                      std::conditional_t<
                          std::is_pointer_v<typename Container::value_type>,
                          std::remove_pointer_t<typename Container::value_type>,
                          typename Container::value_type>>,
    DataFlow<std::string>> {
    return DataFlow<std::string>(
        [&container](std::function<void(const std::string&)> consumer) {
            for (auto& streamHolder : container) {
                std::istream* is_ptr = nullptr;
                if constexpr (std::is_pointer_v<typename Container::value_type>)
                    is_ptr = streamHolder;
                else
                    is_ptr = &streamHolder;
                if (!is_ptr) continue;
                std::string line;
                while (std::getline(*is_ptr, line)) {
                    consumer(line);
                }
                is_ptr->clear();
                is_ptr->seekg(0, std::ios::beg);
            }
        });
}

struct AsVectorWrapper {
    template <typename T>
    std::vector<T> operator()(const DataFlow<T>& input) const {
        std::vector<T> vec;
        input.for_each([&vec](const T& item) { vec.push_back(item); });
        return vec;
    }
};

inline auto AsVector() { return AsVectorWrapper(); }

template <typename Base, typename Joined>
struct JoinResult {
    Base base;
    std::optional<Joined> joined;
    auto operator<=>(const JoinResult&) const = default;
};

template <typename T>
auto get_value(const T& obj) {
    if constexpr (requires { obj.value; })
        return std::decay_t<decltype(obj.value)>(obj.value);
    else
        return std::decay_t<T>(obj);
}

template <typename Base, typename Joined>
bool operator==(const JoinResult<Base, Joined>& lhs,
                const JoinResult<Base, Joined>& rhs) {
    auto eq = [](const auto& a, const auto& b) {
        return get_value(a) == get_value(b);
    };
    auto eq_opt = [&eq](const std::optional<Joined>& a,
                        const std::optional<Joined>& b) {
        if (a.has_value() != b.has_value()) return false;
        if (!a.has_value()) return true;
        return eq(*a, *b);
    };
    return eq(lhs.base, rhs.base) && eq_opt(lhs.joined, rhs.joined);
}

template <typename RightT, typename LeftKeyFunc, typename RightKeyFunc,
          typename CombineFunc>
struct JoinAdapter {
    DataFlow<RightT> right_flow;
    LeftKeyFunc left_key;
    RightKeyFunc right_key;
    CombineFunc combine;

    JoinAdapter(DataFlow<RightT> r, LeftKeyFunc lk, RightKeyFunc rk,
                CombineFunc c)
        : right_flow(r), left_key(lk), right_key(rk), combine(c) {}

    template <typename LeftT>
    auto operator()(const DataFlow<LeftT>& left_flow) const {
        using KeyType =
            std::decay_t<decltype(right_key(*std::declval<RightT*>()))>;
        using ResultType = std::decay_t<decltype(combine(
            std::declval<const LeftT&>(), std::optional<RightT>{}))>;

        std::unordered_multimap<KeyType, RightT> right_map;
        right_flow.for_each([this, &right_map](const RightT& r) {
            right_map.emplace(right_key(r), r);
        });

        auto local_left_key = left_key;
        auto local_combine = combine;

        auto right_map_copy = right_map;

        return DataFlow<ResultType>(
            [left_flow, right_map_copy, local_left_key,
             local_combine](std::function<void(const ResultType&)> consumer) {
                left_flow.for_each([right_map_copy, local_left_key,
                                    local_combine,
                                    &consumer](const LeftT& left_item) {
                    KeyType key = local_left_key(left_item);
                    auto range = right_map_copy.equal_range(key);
                    bool found = false;
                    for (auto it = range.first; it != range.second; ++it) {
                        consumer(local_combine(left_item, it->second));
                        found = true;
                    }
                    if (!found)
                        consumer(
                            local_combine(left_item, std::optional<RightT>{}));
                });
            });
    }
};

template <typename Key, typename VRight, typename LeftKeyFunc,
          typename RightKeyFunc, typename CombineFunc>
struct JoinAdapter<KV<Key, VRight>, LeftKeyFunc, RightKeyFunc, CombineFunc> {
    DataFlow<KV<Key, VRight>> right_flow;
    LeftKeyFunc left_key;
    RightKeyFunc right_key;
    CombineFunc combine;

    JoinAdapter(DataFlow<KV<Key, VRight>> r, LeftKeyFunc lk, RightKeyFunc rk,
                CombineFunc c)
        : right_flow(r), left_key(lk), right_key(rk), combine(c) {}

    template <typename LeftT>
    auto operator()(const DataFlow<LeftT>& left_flow) const {
        std::unordered_multimap<Key, KV<Key, VRight>> right_map;
        right_flow.for_each([&right_map](const KV<Key, VRight>& r) {
            right_map.emplace(r.key, r);
        });

        auto local_combine = combine;
        auto right_map_copy = right_map;

        return DataFlow<decltype(local_combine(
            std::declval<const LeftT&>(), std::optional<KV<Key, VRight>>{}))>(
            [left_flow, right_map_copy,
             local_combine](std::function<void(const auto&)> consumer) {
                left_flow.for_each([right_map_copy, local_combine,
                                    &consumer](const LeftT& left_item) {
                    Key key = left_item.key;
                    auto range = right_map_copy.equal_range(key);
                    bool found = false;
                    for (auto it = range.first; it != range.second; ++it) {
                        consumer(local_combine(left_item, it->second));
                        found = true;
                    }
                    if (!found) {
                        consumer(local_combine(
                            left_item, std::optional<KV<Key, VRight>>{}));
                    }
                });
            });
    }
};

template <typename RightT, typename LeftKeyFunc, typename RightKeyFunc,
          typename CombineFunc>
auto Join(DataFlow<RightT> right_flow, LeftKeyFunc left_key,
          RightKeyFunc right_key, CombineFunc combine) {
    return [=](const auto& left_flow) {
        return JoinAdapter<RightT, LeftKeyFunc, RightKeyFunc, CombineFunc>{
            right_flow, left_key, right_key, combine}(left_flow);
    };
}

template <typename RightT>
auto Join(DataFlow<RightT> right_flow) {
    return [=](const auto& left_flow) {
        using LeftT = typename std::decay_t<decltype(left_flow)>::value_type;
        using LeftValType = decltype(get_value(std::declval<LeftT>()));
        using RightValType = decltype(get_value(std::declval<RightT>()));
        return JoinAdapter<RightT, std::function<LeftT(const LeftT&)>,
                           std::function<RightT(const RightT&)>,
                           std::function<JoinResult<LeftValType, RightValType>(
                               const LeftT&, const std::optional<RightT>&)>>(
            right_flow, std::function<LeftT(const LeftT&)>([](const LeftT& x) {
                return x;
            }),
            std::function<RightT(const RightT&)>(
                [](const RightT& y) { return y; }),
            std::function<JoinResult<LeftValType, RightValType>(
                const LeftT&, const std::optional<RightT>&)>(
                [](const LeftT& l, const std::optional<RightT>& r) {
                    LeftValType left_val = get_value(l);
                    if (r.has_value()) {
                        RightValType right_val = get_value(*r);
                        return JoinResult<LeftValType, RightValType>{left_val,
                                                                     right_val};
                    } else {
                        return JoinResult<LeftValType, RightValType>{
                            left_val, std::nullopt};
                    }
                }))(left_flow);
    };
}

template <typename RightT, typename LeftKeyFunc, typename RightKeyFunc>
auto Join(DataFlow<RightT> right_flow, LeftKeyFunc left_key,
          RightKeyFunc right_key) {
    return [=](const auto& left_flow) {
        using LeftT = typename std::decay_t<decltype(left_flow)>::value_type;
        using RightValType = decltype(get_value(*std::declval<RightT*>()));
        auto combine = [=](const LeftT& l, const std::optional<RightT>& r) {
            auto left_val = get_value(l);
            if (r.has_value()) {
                auto right_val = get_value(*r);
                return JoinResult<decltype(left_val), RightValType>{left_val,
                                                                    right_val};
            } else {
                return JoinResult<decltype(left_val), RightValType>{
                    left_val, std::nullopt};
            }
        };
        return JoinAdapter<RightT, LeftKeyFunc, RightKeyFunc,
                           decltype(combine)>{right_flow, left_key, right_key,
                                              combine}(left_flow);
    };
}

template <typename T>
struct DropNulloptAdapter {
    DataFlow<std::optional<T>> input_flow;
    explicit DropNulloptAdapter(DataFlow<std::optional<T>> flow)
        : input_flow(flow) {}

    DataFlow<T> operator()() const {
        auto flow_copy = input_flow;
        return DataFlow<T>([flow_copy](std::function<void(const T&)> consumer) {
            flow_copy.for_each([consumer](const std::optional<T>& opt) {
                if (opt.has_value()) {
                    consumer(*opt);
                }
            });
        });
    }
};

struct DropNulloptWrapper {
    template <typename T>
    DataFlow<T> operator()(const DataFlow<std::optional<T>>& flow) const {
        return DropNulloptAdapter<T>(flow)();
    }
};

template <typename T>
auto DropNullopt(DataFlow<std::optional<T>> flow) {
    return DropNulloptAdapter<T>(flow)();
}

inline auto DropNullopt() { return DropNulloptWrapper(); }

template <typename T, typename E>
struct ExpectedSplitResult {
    DataFlow<E> unexpected_flow;
    DataFlow<T> expected_flow;
};

struct SplitExpectedWrapper {
    template <typename T, typename E>
    ExpectedSplitResult<T, E> operator()(
        const DataFlow<std::expected<T, E>>& input_flow) const {
        DataFlow<T> expected_flow(
            [input_flow](std::function<void(const T&)> consumer) mutable {
                input_flow.for_each([consumer](const std::expected<T, E>& exp) {
                    if (exp.has_value()) consumer(*exp);
                });
            });
        DataFlow<E> unexpected_flow(
            [input_flow](std::function<void(const E&)> consumer) mutable {
                input_flow.for_each([consumer](const std::expected<T, E>& exp) {
                    if (!exp.has_value()) consumer(exp.error());
                });
            });
        return ExpectedSplitResult<T, E>{unexpected_flow, expected_flow};
    }
};

inline auto SplitExpected() { return SplitExpectedWrapper(); }

template <typename T>
struct is_expected : std::false_type {};

template <typename T, typename E>
struct is_expected<std::expected<T, E>> : std::true_type {};

template <typename F>
auto inline SplitExpected(F f) {
    return [=](const auto& input_flow) {
        using Element = std::decay_t<decltype(*input_flow.begin())>;
        if constexpr (is_expected<Element>::value) {
            return SplitExpected()(input_flow);
        } else {
            return SplitExpected()(input_flow | Transform(f));
        }
    };
}

template <typename Value, typename AggregateFunc, typename KeyFunc>
struct AggregateByKeyWrapper {
    Value initial;
    AggregateFunc aggregate;
    KeyFunc key_func;
    AggregateByKeyWrapper(Value init, AggregateFunc agg, KeyFunc kf)
        : initial(init), aggregate(agg), key_func(kf) {}

    template <typename T>
    auto operator()(const DataFlow<T>& input) const {
        using KeyType = decltype(key_func(std::declval<T>()));
        std::unordered_map<KeyType, Value> map;
        std::vector<KeyType> order;

        input.for_each([this, &map, &order](const T& item) {
            KeyType key = key_func(item);
            if (map.find(key) == map.end()) {
                order.push_back(key);
                map[key] = initial;
            }
            aggregate(item, map[key]);
        });

        return DataFlow<std::pair<KeyType, Value>>(
            [order, map](std::function<void(const std::pair<KeyType, Value>&)>
                             consumer) {
                for (const auto& key : order) {
                    consumer({key, map.at(key)});
                }
            });
    }
};

template <typename Value, typename AggregateFunc, typename KeyFunc>
auto AggregateByKey(Value initial, AggregateFunc agg, KeyFunc kf) {
    return AggregateByKeyWrapper<Value, AggregateFunc, KeyFunc>(initial, agg,
                                                                kf);
}
