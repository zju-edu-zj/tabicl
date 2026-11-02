// Scan centers in ORIGINAL order, retain cdist's scalar accumulation and
// final root, and abandon only when a nonnegative partial sum cannot win.
// No fast-math, changed p, dimension cap, or nearest-center approximation.
#include <ATen/ATen.h>
#include <vector>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/library.h>
#include <cmath>
#include <limits>
#include <numeric>

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

// Complete EM distance matrix: SIMD across centers. exp(p*log(abs(d)))
// evaluates the same power with ordinary rounding differences. Keep the
// historical scalar nearest kernel above for the reference execution mode.
at::Tensor cdist_vector_cpu(const at::Tensor& input, const at::Tensor& centers,
                           double exponent) {
  TORCH_CHECK(input.device().is_cpu() && centers.device().is_cpu());
  TORCH_CHECK(input.dim() == 2 && centers.dim() == 2);
  TORCH_CHECK(input.scalar_type() == centers.scalar_type());
  TORCH_CHECK(input.size(1) == centers.size(1));
  TORCH_CHECK(exponent >= .5 && exponent <= 4.);
  const auto rows = input.size(0), count = centers.size(0), width = input.size(1);
  auto x = input.contiguous(), ct = centers.t().contiguous();
  auto result = at::empty({rows, count}, x.options());
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tabicl_distance_vector_cpu", [&] {
    using Vec = at::vec::Vectorized<scalar_t>;
    const auto* xp = x.const_data_ptr<scalar_t>();
    const auto* cp = ct.const_data_ptr<scalar_t>();
    auto* distances = result.data_ptr<scalar_t>();
    const scalar_t p = scalar_t(exponent), inverse_p = scalar_t(1) / p;
    at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        int64_t center = 0;
        for (; center + Vec::size() <= count; center += Vec::size()) {
          Vec sum(scalar_t(0));
          for (int64_t dim = 0; dim < width; ++dim) {
            Vec diff = (Vec(xp[row * width + dim]) - Vec::loadu(cp + dim * count + center)).abs();
            sum = sum + (p == scalar_t(1) ? diff :
                         p == scalar_t(2) ? diff * diff : (diff.log() * Vec(p)).exp());
          }
          if (p != scalar_t(1)) sum = (sum.log() * Vec(inverse_p)).exp();
          sum.store(distances + row * count + center);
        }
        for (; center < count; ++center) {
          scalar_t sum = 0;
          for (int64_t dim = 0; dim < width; ++dim) {
            const auto d = std::abs(xp[row * width + dim] - cp[dim * count + center]);
            sum += p == scalar_t(1) ? d : p == scalar_t(2) ? d * d : std::pow(d, p);
          }
          distances[row * count + center] = p == scalar_t(1) ? sum : std::pow(sum, inverse_p);
        }
      }
    });
  });
  return result;
}

// SIMD across independent rows also handles small center counts efficiently.
// All terms are nonnegative: a block may stop once no row can improve its
// current winner. No dimension/center is removed from the metric.
at::Tensor nearest_rows_cpu(const at::Tensor& input, const at::Tensor& centers,
                            double exponent) {
  TORCH_CHECK(input.device().is_cpu() && centers.device().is_cpu());
  TORCH_CHECK(input.dim() == 2 && centers.dim() == 2);
  TORCH_CHECK(input.scalar_type() == centers.scalar_type());
  TORCH_CHECK(input.size(1) == centers.size(1) && centers.size(0) > 0);
  TORCH_CHECK(exponent >= .5 && exponent <= 4.);
  const auto rows = input.size(0), count = centers.size(0), width = input.size(1);
  const auto xt = input.t().contiguous(), c = centers.contiguous();
  auto result = at::empty({rows}, input.options().dtype(at::kLong));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tabicl_nearest_rows_cpu", [&] {
    using Vec = at::vec::Vectorized<scalar_t>;
    const auto* xp = xt.const_data_ptr<scalar_t>();
    const auto* cp = c.const_data_ptr<scalar_t>();
    auto* output = result.data_ptr<int64_t>();
    const scalar_t p = scalar_t(exponent);
    // Visit discriminative coordinates first. This only permutes the same
    // nonnegative distance terms and makes the exact lower bound useful sooner.
    std::vector<int64_t> order(width);
    std::iota(order.begin(), order.end(), 0);
    std::vector<scalar_t> ranges(width);
    for (int64_t dim = 0; dim < width; ++dim) {
      scalar_t low = cp[dim], high = cp[dim];
      for (int64_t k = 1; k < count; ++k) {
        low = std::min(low, cp[k*width+dim]);
        high = std::max(high, cp[k*width+dim]);
      }
      ranges[dim] = high-low;
    }
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return ranges[a] > ranges[b]; });
    at::parallel_for(0, (rows + Vec::size()-1)/Vec::size(), 1, [&](int64_t begin, int64_t end) {
      for (auto block = begin; block < end; ++block) {
        const auto row = block * Vec::size(), n = std::min(int64_t(Vec::size()), rows-row);
        Vec best(std::numeric_limits<scalar_t>::infinity()), index(scalar_t(0));
        for (int64_t center = 0; center < count; ++center) {
          Vec sum(scalar_t(0));
          for (auto dim : order) {
            auto d = (Vec::loadu(xp + dim*rows + row, n) - Vec(cp[center*width+dim])).abs();
            sum = sum + (p == scalar_t(1) ? d : p == scalar_t(2) ? d*d : (d.log()*Vec(p)).exp());
            if ((sum < best).zero_mask() == (1 << Vec::size())-1) break;
          }
          const auto improve = sum < best;
          index = Vec::blendv(index, Vec(scalar_t(center)), improve);
          best = at::vec::minimum(best, sum);
        }
        alignas(64) scalar_t indices[Vec::size()];
        index.store(indices);
        for (int64_t k = 0; k < n; ++k) output[row+k] = int64_t(indices[k]);
      }
    });
  });
  return result;
}

at::Tensor quadratic_features_cpu(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu() && input.dim() == 2);
  const auto x = input.contiguous();
  const auto width = x.size(1), basis = width * (width + 1) / 2;
  auto result = at::empty({x.size(0), basis}, x.options());
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tabicl_quadratic_features_cpu", [&] {
    const auto* xp = x.const_data_ptr<scalar_t>();
    auto* rp = result.data_ptr<scalar_t>();
    at::parallel_for(0, x.size(0), 1, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        int64_t k = 0;
        for (int64_t i = 0; i < width; ++i)
          for (int64_t j = i; j < width; ++j)
            rp[row * basis + k++] = xp[row * width + i] * xp[row * width + j];
      }
    });
  });
  return result;
}

// Full EM posterior, vectorized over independent rows (including K < SIMD width).
// ||x-c||_p^q / sigma^q = (sum |x-c|^p)^(q/p) * sigma^(-q).
// No N*K distance/logit intermediates, no discarded centers or dimensions.
at::Tensor em_probabilities_cpu(const at::Tensor& input, const at::Tensor& centers,
                               const at::Tensor& inverse_std_q, const at::Tensor& constants,
                               double exponent, double outer_exponent) {
  TORCH_CHECK(input.device().is_cpu() && centers.device().is_cpu() &&
              inverse_std_q.device().is_cpu() && constants.device().is_cpu());
  TORCH_CHECK(input.dim() == 2 && centers.dim() == 2 && input.size(1) == centers.size(1));
  TORCH_CHECK(input.scalar_type() == centers.scalar_type() &&
              input.scalar_type() == inverse_std_q.scalar_type() && input.scalar_type() == constants.scalar_type());
  const auto rows = input.size(0), count = centers.size(0), width = input.size(1);
  TORCH_CHECK(count > 0 && inverse_std_q.numel() == count && constants.numel() == count);
  TORCH_CHECK(exponent >= .5 && exponent <= 4. && outer_exponent > 0.);
  const auto xt = input.t().contiguous(), c = centers.contiguous();
  const auto scale = inverse_std_q.contiguous(), bias = constants.contiguous();
  auto result = at::empty({rows, count}, input.options());
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tabicl_em_probabilities_cpu", [&] {
    using Vec = at::vec::Vectorized<scalar_t>;
    const auto* xp = xt.const_data_ptr<scalar_t>();
    const auto* cp = c.const_data_ptr<scalar_t>();
    const auto* sp = scale.const_data_ptr<scalar_t>();
    const auto* bp = bias.const_data_ptr<scalar_t>();
    auto* output = result.data_ptr<scalar_t>();
    const scalar_t p = scalar_t(exponent), ratio = scalar_t(outer_exponent / exponent);
    at::parallel_for(0, (rows+Vec::size()-1)/Vec::size(), 1, [&](int64_t begin, int64_t end) {
      std::vector<Vec> logits(count);
      for (auto block = begin; block < end; ++block) {
        const auto row = block*Vec::size(), n = std::min(int64_t(Vec::size()), rows-row);
        Vec maximum(-std::numeric_limits<scalar_t>::infinity());
        for (int64_t center = 0; center < count; ++center) {
          Vec sum(scalar_t(0));
          for (int64_t dim = 0; dim < width; ++dim) {
            const auto diff = (Vec::loadu(xp+dim*rows+row, n) - Vec(cp[center*width+dim])).abs();
            sum = sum + (p == scalar_t(1) ? diff : p == scalar_t(2) ? diff*diff : (diff.log()*Vec(p)).exp());
          }
          const auto power = ratio == scalar_t(1) ? sum : (sum.log()*Vec(ratio)).exp();
          logits[center] = Vec(bp[center]) - power*Vec(sp[center]);
          maximum = at::vec::maximum(maximum, logits[center]);
        }
        Vec total(scalar_t(0));
        for (auto& logit : logits) {
          logit = (logit-maximum).exp();
          total = total+logit;
        }
        for (int64_t center = 0; center < count; ++center) {
          alignas(64) scalar_t values[Vec::size()];
          (logits[center]/total).store(values);
          for (int64_t j = 0; j < n; ++j) output[(row+j)*count+center] = values[j];
        }
      }
    });
  });
  return result;
}

TORCH_LIBRARY(tabicl_prior_exact, m) {
  m.def("nearest_cpu(Tensor x, Tensor centers, float p) -> Tensor");
  m.def("tree_cpu(Tensor x, Tensor dimensions, Tensor thresholds, Tensor leaves) -> Tensor");
  m.def("nearest_vector_cpu(Tensor x, Tensor centers, float p) -> Tensor");
  m.def("cdist_vector_cpu(Tensor x, Tensor centers, float p) -> Tensor");
  m.def("quadratic_features_cpu(Tensor x) -> Tensor");
  m.def("em_probabilities_cpu(Tensor x, Tensor centers, Tensor inverse_std_q, Tensor constants, float p, float q) -> Tensor");
}
TORCH_LIBRARY_IMPL(tabicl_prior_exact, CPU, m) {
  m.impl("nearest_cpu", &nearest_cpu);
  m.impl("tree_cpu", &tree_cpu);
  m.impl("nearest_vector_cpu", &nearest_rows_cpu);
  m.impl("cdist_vector_cpu", &cdist_vector_cpu);
  m.impl("quadratic_features_cpu", &quadratic_features_cpu);
  m.impl("em_probabilities_cpu", &em_probabilities_cpu);
}
