// Scan centers in ORIGINAL order, retain cdist's scalar accumulation and
// final root, and abandon only when a nonnegative partial sum cannot win.
// No fast-math, changed p, dimension cap, or nearest-center approximation.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <cmath>
#include <limits>

at::Tensor nearest_cpu(const at::Tensor& input, const at::Tensor& centers,
                       double exponent) {
  TORCH_CHECK(input.device().is_cpu() && centers.device().is_cpu());
  TORCH_CHECK(input.dim() == 2 && centers.dim() == 2);
  TORCH_CHECK(input.scalar_type() == centers.scalar_type());
  TORCH_CHECK(input.size(1) == centers.size(1) && centers.size(0) > 0);
  TORCH_CHECK(exponent >= .5 && exponent <= 4. && exponent != 2.);
  auto x = input.contiguous(), c = centers.contiguous();
  auto result = at::empty({x.size(0)}, x.options().dtype(at::kLong));
  const auto rows = x.size(0), count = c.size(0), width = x.size(1);
  auto* output = result.data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tabicl_nearest_cpu", [&] {
    const auto* xp = x.const_data_ptr<scalar_t>();
    const auto* cp = c.const_data_ptr<scalar_t>();
    const scalar_t p = static_cast<scalar_t>(exponent);
    if (p == scalar_t(2)) {
      result = at::cdist(input, centers, exponent).argmin(-1);
      return;
    }
    at::parallel_for(0, rows, 1, [&](int64_t start, int64_t end) {
      for (int64_t row = start; row < end; ++row) {
        scalar_t best_sum = std::numeric_limits<scalar_t>::infinity();
        scalar_t best_distance = std::numeric_limits<scalar_t>::infinity();
        int64_t best_index = 0;
        for (int64_t center = 0; center < count; ++center) {
          scalar_t sum = 0;
          bool abandoned = false;
          for (int64_t dim = 0; dim < width; ++dim) {
            const scalar_t difference = std::abs(xp[row*width+dim] - cp[center*width+dim]);
            sum += p == scalar_t(1) ? difference : std::pow(difference, p);
            if (sum >= best_sum) {
              abandoned = true;
              break;
            }
          }
          if (abandoned) continue;
          const scalar_t distance = p == scalar_t(1) ? sum : std::pow(sum, 1.0 / p);
          // Comparing the SUM alone would break ties created by the root.
          if (distance < best_distance) {
            best_sum = sum;
            best_distance = distance;
            best_index = center;
          }
        }
        output[row] = best_index;
      }
    });
  });
  return result;
}

// Evaluate every tree and every output, without materializing [rows, trees,
// depth] split tensors or [rows, trees, outputs] gathered leaf tensors.
at::Tensor tree_cpu(const at::Tensor& input, const at::Tensor& dimensions,
                    const at::Tensor& thresholds, const at::Tensor& leaves) {
  TORCH_CHECK(input.device().is_cpu() && dimensions.device().is_cpu() &&
              thresholds.device().is_cpu() && leaves.device().is_cpu());
  TORCH_CHECK(input.dim() == 2 && leaves.dim() == 3 && dimensions.dim() == 1 &&
              thresholds.dim() == 1 && dimensions.scalar_type() == at::kLong);
  TORCH_CHECK(input.scalar_type() == thresholds.scalar_type() &&
              input.scalar_type() == leaves.scalar_type());
  const auto trees = leaves.size(0), count = leaves.size(1), outputs = leaves.size(2);
  TORCH_CHECK(trees > 0 && count >= 2 && (count & (count - 1)) == 0);
  int64_t depth = 0;
  for (auto value = count; value > 1; value >>= 1) ++depth;
  TORCH_CHECK(dimensions.numel() == trees * depth && thresholds.numel() == trees * depth);
  TORCH_CHECK(dimensions.min().item<int64_t>() >= 0 &&
              dimensions.max().item<int64_t>() < input.size(1));
  const auto x = input.contiguous(), dims = dimensions.contiguous();
  const auto splits = thresholds.contiguous(), values = leaves.contiguous();
  auto result = at::zeros({x.size(0), outputs}, x.options());
  const auto* dp = dims.const_data_ptr<int64_t>();
  const auto width = x.size(1);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tabicl_tree_cpu", [&] {
    const auto* xp = x.const_data_ptr<scalar_t>();
    const auto* sp = splits.const_data_ptr<scalar_t>();
    const auto* vp = values.const_data_ptr<scalar_t>();
    auto* rp = result.data_ptr<scalar_t>();
    at::parallel_for(0, x.size(0), 1, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        auto* output = rp + row * outputs;
        for (int64_t tree = 0; tree < trees; ++tree) {
          int64_t leaf = 0;
          for (int64_t level = 0; level < depth; ++level) {
            const auto split = tree * depth + level;
            leaf |= int64_t(xp[row * width + dp[split]] > sp[split]) << level;
          }
          const auto* value = vp + (tree * count + leaf) * outputs;
          for (int64_t feature = 0; feature < outputs; ++feature) output[feature] += value[feature];
        }
        for (int64_t feature = 0; feature < outputs; ++feature) output[feature] /= scalar_t(trees);
      }
    });
  });
  return result;
}

TORCH_LIBRARY(tabicl_prior_exact, m) {
  m.def("nearest_cpu(Tensor x, Tensor centers, float p) -> Tensor");
  m.def("tree_cpu(Tensor x, Tensor dimensions, Tensor thresholds, Tensor leaves) -> Tensor");
}
TORCH_LIBRARY_IMPL(tabicl_prior_exact, CPU, m) {
  m.impl("nearest_cpu", &nearest_cpu);
  m.impl("tree_cpu", &tree_cpu);
}
