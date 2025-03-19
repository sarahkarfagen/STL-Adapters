#pragma once

#include <optional>

template <typename Key, typename Value>
struct KV {
	Key key;
	Value value;
};

template <typename Base, typename Joined>
struct JoinResult {
	Base base;
	std::optional<Joined> joined;
};
