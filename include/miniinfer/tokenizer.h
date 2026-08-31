#pragma once
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace miniinfer {
class Tokenizer {
 public:
  void add_token(int id, std::string token);
  void add_merge(std::string left, std::string right);
  void add_special_token(int id, std::string token);
  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  bool load(const std::string& path);
  const std::unordered_map<int, std::string>& vocabulary() const { return id_to_token_; }
  const std::vector<std::pair<std::string, std::string>>& merges() const { return merges_; }
  const std::vector<std::pair<int, std::string>>& special_tokens() const { return special_tokens_; }
  void set_special_ids(int bos, int eos) { bos_id_ = bos; eos_id_ = eos; }
  int bos_id() const { return bos_id_; }
  int eos_id() const { return eos_id_; }
 private:
  std::unordered_map<int, std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::pair<std::string, std::string>> merges_;
  std::unordered_map<std::string, size_t> merge_rank_;
  std::vector<std::pair<int, std::string>> special_tokens_;
  int bos_id_ = -1, eos_id_ = -1;
};
}
