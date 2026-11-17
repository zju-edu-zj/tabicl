// Full bootstrap ExtraTrees for the prior filter. Candidate evaluation never
// partitions samples; only the selected split writes the sample permutation.
// No fitted state is shared across tables or trees. All arithmetic is double
// except X/extrema, matching the float32 feature contract of sklearn trees.
#pragma once
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <vector>

namespace tabicl_forest {
struct Node {
  int feature = -1, left = -1, right = -1, samples = 0;
  double threshold = 0., weight = 0.;
  std::vector<double> value;
};

// Same xorshift family/range used by sklearn's local splitter RNG. Changing
// sample order can change ties; this is a distribution contract, not bitwise
// reproduction of sklearn's traversal or random-number consumption.
struct Random {
  uint32_t state;
  explicit Random(int64_t seed): state(uint32_t(seed)) {}
  uint32_t next() {
    if (!state) state = 1;
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state % (uint32_t(0x7fffffff) + 1U);
  }
  int integer(int high) { return int(next() % uint32_t(high)); }
  double uniform(double low, double high) {
    return low + (high-low) * (double(next()) / double(0x7fffffff));
  }
};

struct Tree {
  const float* x;  // feature-major, copied once for the entire forest
  const double* y;
  const double* weights;
  int n, d, outputs, classes, width, max_depth;
  Random random;
  std::vector<int> rows;
  std::vector<Node> nodes;
  double root_weight = 0.;

  Tree(const float* features, const double* targets, const double* w,
       int nr, int nc, int no, int cl, int depth, int64_t seed):
      x(features), y(targets), weights(w), n(nr), d(nc), outputs(no),
      classes(cl), width(cl ? cl : no), max_depth(depth), random(seed) {
    nodes.reserve((1 << (max_depth+1))-1);
    for (int i=0; i<n; ++i) if (weights[i] > 0.) {
      rows.push_back(i); root_weight += weights[i];
    }
    build(0, int(rows.size()), 0, {});
  }

  int build(int begin, int end, int depth, const std::vector<int>& inherited_constants) {
    const int id = int(nodes.size());
    nodes.emplace_back();
    Node node;
    node.samples = end-begin;
    node.value.assign(width, 0.);
    double squared = 0.;
    for (int position=begin; position<end; ++position) {
      const int row = rows[position]; const double w = weights[row];
      node.weight += w;
      if (classes) node.value[int(y[row])] += w;
      else for (int j=0; j<outputs; ++j) {
        const double target = y[row*outputs+j];
        node.value[j] += w*target; squared += w*target*target;
      }
    }
    double norm = 0.;
    for (auto value: node.value) norm += value*value;
    const double impurity = classes ? 1.-norm/(node.weight*node.weight)
      : (squared/node.weight-norm/(node.weight*node.weight))/outputs;
    auto totals = node.value;
    for (auto& value: node.value) value /= node.weight;
    nodes[id] = node;
    if (depth >= max_depth || end-begin < 2 || impurity <= DBL_EPSILON) return id;

    std::vector<int> features;
    std::vector<int> constants = inherited_constants;
    std::vector<uint8_t> is_constant(d, 0);
    for (int j: constants) is_constant[j] = 1;
    for (int j=0; j<d; ++j) if (!is_constant[j]) features.push_back(j);
    std::vector<float> values(end-begin);
    std::vector<double> left(width);
    double best_score = -std::numeric_limits<double>::infinity();
    int best_feature = -1;
    double best_threshold = 0.;
    // Uniform feature order, all nonconstant features examined. Constant
    // discoveries propagate to BOTH descendants, as in the full forest.
    for (int remaining=int(features.size()); remaining>0; --remaining) {
      const int selected = random.integer(remaining);
      const int feature = features[selected];
      std::swap(features[selected], features[remaining-1]);
      const float* column = x + int64_t(feature)*n;
      float low = std::numeric_limits<float>::infinity(), high = -low;
      for (int position=begin; position<end; ++position) {
        const float value = column[rows[position]];
        values[position-begin] = value;
        low = std::min(low, value); high = std::max(high, value);
      }
      if (high <= low + 1.e-7f) { constants.push_back(feature); continue; }
      double threshold = random.uniform(low, high);
      if (threshold == high) threshold = low;
      std::fill(left.begin(), left.end(), 0.);
      double left_weight = 0.; int left_count = 0;
      for (int position=begin; position<end; ++position) {
        if (values[position-begin] <= threshold) {
          const int row = rows[position]; const double w = weights[row];
          left_weight += w; ++left_count;
          if (classes) left[int(y[row])] += w;
          else for (int j=0; j<outputs; ++j) left[j] += w*y[row*outputs+j];
        }
      }
      if (left_count == 0 || left_count == end-begin) continue;
      const double right_weight = node.weight-left_weight;
      double score = 0.;
      for (int j=0; j<width; ++j) {
        const double right = totals[j]-left[j];
        score += left[j]*left[j]/left_weight + right*right/right_weight;
      }
      if (score > best_score) {
        best_score = score; best_feature = feature; best_threshold = threshold;
      }
    }
    // Equivalent min_impurity_decrease=0 test, including numerical epsilon.
    const double improvement = (best_score-norm/node.weight)/root_weight/(classes ? 1 : outputs);
    if (best_feature < 0 || improvement < -DBL_EPSILON) return id;
    const float* column = x + int64_t(best_feature)*n;
    auto middle = std::partition(rows.begin()+begin, rows.begin()+end,
      [&](int row) { return column[row] <= best_threshold; });
    const int split = int(middle-rows.begin());
    nodes[id].feature = best_feature; nodes[id].threshold = best_threshold;
    nodes[id].left = build(begin, split, depth+1, constants);
    nodes[id].right = build(split, end, depth+1, constants);
    return id;
  }

  const std::vector<double>& predict(int row) const {
    int id = 0;
    while (nodes[id].feature >= 0) {
      const auto& node = nodes[id];
      id = x[int64_t(node.feature)*n+row] <= node.threshold ? node.left : node.right;
    }
    return nodes[id].value;
  }
};

inline void validate(const at::Tensor& x, const at::Tensor& y,
                     const at::Tensor& weights, int64_t classes, int64_t depth) {
  TORCH_CHECK(x.device().is_cpu() && y.device().is_cpu() && weights.device().is_cpu());
  TORCH_CHECK(x.scalar_type()==at::kFloat && y.scalar_type()==at::kDouble && weights.scalar_type()==at::kDouble);
  TORCH_CHECK(x.dim()==2 && y.dim()==2 && weights.dim()==2);
  TORCH_CHECK(x.size(0)>=2 && x.size(0)<=65536 && x.size(1)>0 && x.size(1)<=4096);
  TORCH_CHECK(y.size(0)==x.size(0) && y.size(1)>0 && y.size(1)<=256);
  TORCH_CHECK(weights.size(1)==x.size(0) && weights.size(0)>0 && weights.size(0)<=256);
  TORCH_CHECK(classes>=0 && classes<=256 && depth>=0 && depth<=12);
  TORCH_CHECK(at::isfinite(x).all().item<bool>() && at::isfinite(y).all().item<bool>());
  TORCH_CHECK(at::isfinite(weights).all().item<bool>() && (weights>=0).all().item<bool>());
  TORCH_CHECK((weights.sum(1)>0).all().item<bool>());
  if (classes) {
    TORCH_CHECK(y.size(1)==1 && classes>=2);
    TORCH_CHECK((y>=0).all().item<bool>() && (y<classes).all().item<bool>());
    TORCH_CHECK((y==y.floor()).all().item<bool>());
  }
}
} // namespace tabicl_forest

std::vector<at::Tensor> forest_tree_cpu(const at::Tensor& input, const at::Tensor& targets,
    const at::Tensor& weight, int64_t seed, int64_t classes, int64_t depth) {
  auto weights = weight.reshape({1, -1});
  tabicl_forest::validate(input, targets, weights, classes, depth);
  auto x = input.t().contiguous(), y = targets.contiguous(), w = weights.contiguous();
  tabicl_forest::Tree tree(x.const_data_ptr<float>(), y.const_data_ptr<double>(), w.const_data_ptr<double>(),
    input.size(0), input.size(1), y.size(1), classes, depth, seed);
  auto metadata = at::empty({int64_t(tree.nodes.size()), 6}, y.options());
  auto values = at::empty({int64_t(tree.nodes.size()), tree.width}, y.options());
  auto* mp = metadata.data_ptr<double>(); auto* vp = values.data_ptr<double>();
  for (size_t i=0; i<tree.nodes.size(); ++i) {
    const auto& node = tree.nodes[i];
    mp[i*6] = node.feature; mp[i*6+1] = node.threshold;
    mp[i*6+2] = node.left; mp[i*6+3] = node.right;
    mp[i*6+4] = node.samples; mp[i*6+5] = node.weight;
    std::copy(node.value.begin(), node.value.end(), vp+i*tree.width);
  }
  return {metadata, values};
}

at::Tensor forest_oob_cpu(const at::Tensor& input, const at::Tensor& targets,
    const at::Tensor& weights, const at::Tensor& seeds, int64_t classes, int64_t depth) {
  tabicl_forest::validate(input, targets, weights, classes, depth);
  TORCH_CHECK(seeds.device().is_cpu() && seeds.scalar_type()==at::kLong
    && seeds.dim()==1 && seeds.size(0)==weights.size(0));
  auto x = input.t().contiguous(), y = targets.contiguous(), w = weights.contiguous(), s = seeds.contiguous();
  const int n = input.size(0), d = input.size(1), outputs = y.size(1), width = classes ? classes : outputs;
  auto result = at::zeros({n, width}, y.options());
  auto* prediction = result.data_ptr<double>();
  std::vector<int> votes(n, 0);
  for (int64_t t=0; t<weights.size(0); ++t) {
    const auto* weight = w.const_data_ptr<double>()+t*n;
    tabicl_forest::Tree tree(x.const_data_ptr<float>(), y.const_data_ptr<double>(), weight,
      n, d, outputs, classes, depth, s.const_data_ptr<int64_t>()[t]);
    for (int row=0; row<n; ++row) if (weight[row] == 0.) {
      const auto& value = tree.predict(row); ++votes[row];
      for (int j=0; j<width; ++j) prediction[row*width+j] += value[j];
    }
  }
  for (int row=0; row<n; ++row) if (votes[row])
    for (int j=0; j<width; ++j) prediction[row*width+j] /= votes[row];
  return result;
}
