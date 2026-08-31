#include "miniinfer/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <unordered_map>

namespace miniinfer {
namespace {
struct Codepoint {
  uint32_t value;
  size_t begin;
  size_t end;
};

std::string utf8(uint32_t cp) {
  std::string out;
  if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
  else if (cp <= 0x7ff) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp <= 0xffff) {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
  return out;
}

std::vector<Codepoint> codepoints(const std::string& text) {
  std::vector<Codepoint> result;
  for (size_t i = 0; i < text.size();) {
    const size_t begin = i;
    const unsigned char first = static_cast<unsigned char>(text[i++]);
    uint32_t value;
    size_t remaining;
    if (first < 0x80) { value = first; remaining = 0; }
    else if ((first & 0xe0) == 0xc0) { value = first & 0x1f; remaining = 1; }
    else if ((first & 0xf0) == 0xe0) { value = first & 0x0f; remaining = 2; }
    else if ((first & 0xf8) == 0xf0) { value = first & 0x07; remaining = 3; }
    else { value = 0xfffd; remaining = 0; }
    bool valid = true;
    for (size_t j = 0; j < remaining; ++j) {
      if (i >= text.size() || (static_cast<unsigned char>(text[i]) & 0xc0) != 0x80) {
        valid = false; break;
      }
      value = (value << 6) | (static_cast<unsigned char>(text[i++]) & 0x3f);
    }
    if (!valid) { value = 0xfffd; i = begin + 1; }
    result.push_back({value, begin, i});
  }
  return result;
}

struct UnicodeRange { uint32_t first, last; };
#include "unicode_ranges.inc"

template<size_t N>
bool in_ranges(uint32_t cp, const UnicodeRange (&ranges)[N]) {
  size_t left = 0, right = N;
  while (left < right) {
    const size_t middle = left + (right - left) / 2;
    if (cp < ranges[middle].first) right = middle;
    else if (cp > ranges[middle].last) left = middle + 1;
    else return true;
  }
  return false;
}

bool is_space(uint32_t cp) { return in_ranges(cp, kWhitespaceRanges); }
bool is_number(uint32_t cp) { return in_ranges(cp, kNumberRanges); }
[[maybe_unused]] bool is_decimal_digit(uint32_t cp) { return in_ranges(cp, kDecimalRanges); }
bool is_letter(uint32_t cp) { return in_ranges(cp, kLetterRanges); }

const std::vector<std::string>& byte_encoder() {
  static const std::vector<std::string> table = [] {
    std::vector<int> bytes;
    for (int c = 33; c <= 126; ++c) bytes.push_back(c);
    for (int c = 161; c <= 172; ++c) bytes.push_back(c);
    for (int c = 174; c <= 255; ++c) bytes.push_back(c);
    std::vector<bool> present(256, false);
    for (int c : bytes) present[c] = true;
    std::vector<int> unicode = bytes;
    int extra = 0;
    for (int c = 0; c < 256; ++c)
      if (!present[c]) { bytes.push_back(c); unicode.push_back(256 + extra++); }
    std::vector<std::string> result(256);
    for (size_t i = 0; i < bytes.size(); ++i) result[bytes[i]] = utf8(unicode[i]);
    return result;
  }();
  return table;
}

const std::unordered_map<std::string, unsigned char>& byte_decoder() {
  static const std::unordered_map<std::string, unsigned char> table = [] {
    std::unordered_map<std::string, unsigned char> result;
    const auto& encoder = byte_encoder();
    for (size_t i = 0; i < encoder.size(); ++i)
      result[encoder[i]] = static_cast<unsigned char>(i);
    return result;
  }();
  return table;
}

std::vector<std::string> split_utf8(const std::string& text) {
  std::vector<std::string> result;
  for (const auto& cp : codepoints(text)) result.push_back(text.substr(cp.begin, cp.end - cp.begin));
  return result;
}

enum class Kind { Space, Letter, Number, Other };
Kind kind_of(uint32_t cp) {
  if (is_space(cp)) return Kind::Space;
  if (is_letter(cp)) return Kind::Letter;
  if (is_number(cp)) return Kind::Number;
  return Kind::Other;
}

// ByteLevel's GPT-2 regex over one input split.
std::vector<std::string> bytelevel_pretokenize(const std::string& text) {
  const auto cps = codepoints(text);
  std::vector<std::string> result;
  size_t i = 0;
  while (i < cps.size()) {
    const size_t begin_i = i;

    // Contractions are higher-priority alternatives in the GPT-2 regex.
    if (cps[i].value == '\'' && i + 1 < cps.size()) {
      static const std::vector<std::string> suffixes = {"s", "t", "re", "ve", "m", "ll", "d"};
      for (const auto& suffix : suffixes) {
        const size_t end_byte = cps[i].begin + 1 + suffix.size();
        if (end_byte <= text.size() && text.compare(cps[i].begin + 1, suffix.size(), suffix) == 0) {
          size_t count = 1 + suffix.size();
          if (i + count <= cps.size()) {
            result.push_back(text.substr(cps[i].begin, cps[i + count - 1].end - cps[i].begin));
            i += count;
            goto next_piece;
          }
        }
      }
    }

    // A single ordinary space can prefix letter, number, or punctuation runs.
    if (cps[i].value == 0x20 && i + 1 < cps.size() && !is_space(cps[i + 1].value)) ++i;
    if (i >= cps.size()) {
      result.push_back(text.substr(cps[begin_i].begin));
      break;
    }

    {
      const Kind kind = kind_of(cps[i].value);
      if (kind == Kind::Space) {
        while (i < cps.size() && kind_of(cps[i].value) == Kind::Space) ++i;
        // Before non-space text, leave one ASCII space for the next regex match.
        if (i < cps.size() && i - begin_i > 1 && cps[i - 1].value == 0x20) --i;
      } else {
        while (i < cps.size() && kind_of(cps[i].value) == kind) ++i;
      }
    }
    result.push_back(text.substr(cps[begin_i].begin, cps[i - 1].end - cps[begin_i].begin));
next_piece:
    continue;
  }
  return result;
}

// SmolLM2 applies Digits(individual_digits=true) before ByteLevel. Tokenizers'
// Digits pre-tokenizer treats every Unicode Number category as a digit split.
std::vector<std::string> pretokenize(const std::string& text) {
  const auto cps = codepoints(text);
  std::vector<std::string> result;
  size_t ordinary_begin = 0;
  for (const auto& cp : cps) {
    if (!is_number(cp.value)) continue;
    if (cp.begin > ordinary_begin) {
      auto pieces = bytelevel_pretokenize(text.substr(ordinary_begin, cp.begin - ordinary_begin));
      result.insert(result.end(), pieces.begin(), pieces.end());
    }
    result.push_back(text.substr(cp.begin, cp.end - cp.begin));
    ordinary_begin = cp.end;
  }
  if (ordinary_begin < text.size()) {
    auto pieces = bytelevel_pretokenize(text.substr(ordinary_begin));
    result.insert(result.end(), pieces.begin(), pieces.end());
  }
  return result;
}

std::string pair_key(const std::string& left, const std::string& right) {
  return left + std::string(1, '\0') + right;
}
}  // namespace

void Tokenizer::add_token(int id, std::string token) {
  token_to_id_[token] = id;
  id_to_token_[id] = std::move(token);
}

void Tokenizer::add_merge(std::string left, std::string right) {
  merge_rank_[pair_key(left, right)] = merges_.size();
  merges_.emplace_back(std::move(left), std::move(right));
}

void Tokenizer::add_special_token(int id, std::string token) {
  add_token(id, token);
  special_tokens_.emplace_back(id, std::move(token));
  std::sort(special_tokens_.begin(), special_tokens_.end(), [](const auto& a, const auto& b) {
    return a.second.size() > b.second.size();
  });
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  const auto encode_ordinary = [&](const std::string& ordinary, std::vector<int>& output) {
    const auto& bytes = byte_encoder();
    for (const std::string& piece : pretokenize(ordinary)) {
      std::string encoded;
      for (unsigned char byte : piece) encoded += bytes[byte];
      std::vector<std::string> symbols = split_utf8(encoded);
      while (symbols.size() > 1) {
        size_t best_rank = std::numeric_limits<size_t>::max(), best_pos = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
          auto it = merge_rank_.find(pair_key(symbols[i], symbols[i + 1]));
          if (it != merge_rank_.end() && it->second < best_rank) {
            best_rank = it->second; best_pos = i;
          }
        }
        if (best_rank == std::numeric_limits<size_t>::max()) break;
        const std::string left = symbols[best_pos], right = symbols[best_pos + 1];
        std::vector<std::string> merged;
        for (size_t i = 0; i < symbols.size();) {
          if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
            merged.push_back(left + right); i += 2;
          } else merged.push_back(symbols[i++]);
        }
        symbols = std::move(merged);
      }
      for (const auto& symbol : symbols) {
        auto it = token_to_id_.find(symbol);
        if (it == token_to_id_.end())
          throw std::runtime_error("BPE token is missing from vocabulary");
        output.push_back(it->second);
      }
    }
  };

  size_t position = 0;
  while (position < text.size()) {
    size_t next = text.size();
    const std::pair<int, std::string>* matched = nullptr;
    for (const auto& special : special_tokens_) {
      size_t found = text.find(special.second, position);
      if (found < next || (found == next && matched && special.second.size() > matched->second.size())) {
        next = found; matched = &special;
      }
    }
    if (next > position) encode_ordinary(text.substr(position, next - position), ids);
    if (!matched) break;
    ids.push_back(matched->first);
    position = next + matched->second.size();
  }
  return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string result, encoded;
  const auto flush = [&] (std::string& bytes_text, std::string& output) {
    const auto& decoder = byte_decoder();
    for (const auto& cp : split_utf8(bytes_text)) {
      auto it = decoder.find(cp);
      if (it != decoder.end()) output.push_back(static_cast<char>(it->second));
      else output += cp;
    }
    bytes_text.clear();
  };
  for (int id : ids) {
    auto it = id_to_token_.find(id);
    if (it == id_to_token_.end()) continue;
    bool special = false;
    for (const auto& item : special_tokens_) if (item.first == id) { special = true; break; }
    if (special) { flush(encoded, result); result += it->second; }
    else encoded += it->second;
  }
  flush(encoded, result);
  return result;
}

bool Tokenizer::load(const std::string& path) {
  std::ifstream file(path);
  if (!file) return false;
  int id;
  std::string token;
  while (file >> id >> token) add_token(id, token);
  return true;
}
}  // namespace miniinfer
