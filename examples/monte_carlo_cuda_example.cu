#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cstdlib>

inline void cuda_check(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(code) << " at " << file << ":" << line
                  << std::endl;
        std::exit(static_cast<int>(code));
    }
}

#define CUDA_CHECK(ans) cuda_check((ans), __FILE__, __LINE__)

namespace
{
constexpr int kGroups = 2;
constexpr int kNx = 6;
constexpr int kNy = 6;
constexpr int kNz = 6;
constexpr int kNumCells = kNx * kNy * kNz;
constexpr int kReactionPerGroup = 3;
constexpr int kReactionCounts = kGroups * kReactionPerGroup + 1;
constexpr float kEpsilon = 1.0e-7f;
constexpr float kPi = 3.14159265358979323846f;

struct Vec3
{
    float x;
    float y;
    float z;
};

struct MultiGroupMaterial
{
    float sigma_abs[kGroups];
    float sigma_f[kGroups];
    float nu[kGroups];
    float chi[kGroups];
    float scatter[kGroups][kGroups];

    __host__ __device__ float scatter_total(int g) const
    {
        float total = 0.0f;
        for (int gp = 0; gp < kGroups; ++gp)
        {
            total += scatter[g][gp];
        }
        return total;
    }

    __host__ __device__ float total_xs(int g) const
    {
        return scatter_total(g) + sigma_abs[g] + sigma_f[g];
    }
};

struct FixedSource
{
    float chi[kGroups];
};

struct ParticleSource
{
    Vec3 position;
    Vec3 direction;
    int group;
    float weight;
};

__device__ inline unsigned long long atomicAdd64(unsigned long long* address,
                                                 unsigned long long val)
{
#if __CUDA_ARCH__ >= 350
    return atomicAdd(address, val);
#else
    unsigned long long old = *address;
    unsigned long long assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, assumed + val);
    } while (assumed != old);
    return old;
#endif
}

__host__ __device__ inline Vec3 make_vec3(float x, float y, float z)
{
    Vec3 v{x, y, z};
    return v;
}

__host__ __device__ inline Vec3 add(const Vec3& a, const Vec3& b)
{
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline Vec3 scale(const Vec3& v, float s)
{
    return make_vec3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline Vec3 mad(const Vec3& a, const Vec3& b, float s)
{
    return make_vec3(a.x + b.x * s, a.y + b.y * s, a.z + b.z * s);
}

__host__ __device__ inline bool inside_domain(const Vec3& p, const Vec3& min, const Vec3& max)
{
    return (p.x >= min.x && p.x < max.x && p.y >= min.y && p.y < max.y && p.z >= min.z && p.z < max.z);
}

__device__ inline Vec3 sample_isotropic_direction(curandState* state)
{
    const float mu = 2.0f * curand_uniform(state) - 1.0f;
    const float phi = 2.0f * kPi * curand_uniform(state);
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));
    return make_vec3(sin_theta * cosf(phi), sin_theta * sinf(phi), mu);
}

__device__ inline float sample_free_path(curandState* state, float sigma_t)
{
    float xi = curand_uniform(state);
    xi = fmaxf(xi, kEpsilon);
    return -logf(xi) / fmaxf(sigma_t, kEpsilon);
}

__device__ inline int sample_group_from_chi(const float* chi, curandState* state)
{
    float xi = curand_uniform(state);
    float cumulative = 0.0f;
    for (int g = 0; g < kGroups; ++g)
    {
        cumulative += chi[g];
        if (xi <= cumulative)
        {
            return g;
        }
    }
    return kGroups - 1;
}

__device__ inline int sample_scatter_group(const MultiGroupMaterial& material,
                                           int g,
                                           float scatter_total,
                                           curandState* state)
{
    if (scatter_total <= 0.0f)
    {
        return g;
    }
    float xi = curand_uniform(state) * scatter_total;
    float cumulative = 0.0f;
    for (int gp = 0; gp < kGroups; ++gp)
    {
        cumulative += material.scatter[g][gp];
        if (xi <= cumulative)
        {
            return gp;
        }
    }
    return kGroups - 1;
}

__device__ inline float distance_to_boundary(const Vec3& position,
                                             const Vec3& direction,
                                             const Vec3& min_bound,
                                             const Vec3& max_bound)
{
    const float huge = 1.0e30f;
    float distance = huge;

    if (fabsf(direction.x) > kEpsilon)
    {
        float tx = (direction.x > 0.0f) ? (max_bound.x - position.x) / direction.x
                                       : (min_bound.x - position.x) / direction.x;
        if (tx > kEpsilon)
        {
            distance = fminf(distance, tx);
        }
    }

    if (fabsf(direction.y) > kEpsilon)
    {
        float ty = (direction.y > 0.0f) ? (max_bound.y - position.y) / direction.y
                                       : (min_bound.y - position.y) / direction.y;
        if (ty > kEpsilon)
        {
            distance = fminf(distance, ty);
        }
    }

    if (fabsf(direction.z) > kEpsilon)
    {
        float tz = (direction.z > 0.0f) ? (max_bound.z - position.z) / direction.z
                                       : (min_bound.z - position.z) / direction.z;
        if (tz > kEpsilon)
        {
            distance = fminf(distance, tz);
        }
    }

    return distance;
}

__device__ inline int cell_index_from_position(const Vec3& position,
                                               const Vec3& min_bound,
                                               float dx,
                                               float dy,
                                               float dz)
{
    const float x_rel = (position.x - min_bound.x) / dx;
    const float y_rel = (position.y - min_bound.y) / dy;
    const float z_rel = (position.z - min_bound.z) / dz;

    int ix = static_cast<int>(floorf(x_rel));
    int iy = static_cast<int>(floorf(y_rel));
    int iz = static_cast<int>(floorf(z_rel));

    ix = max(0, min(ix, kNx - 1));
    iy = max(0, min(iy, kNy - 1));
    iz = max(0, min(iz, kNz - 1));

    return ix + iy * kNx + iz * kNx * kNy;
}

__global__ void initialize_rng(curandState* states,
                               unsigned long long seed,
                               unsigned long long offset,
                               unsigned int histories)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= histories)
    {
        return;
    }
    curand_init(seed, tid, offset, &states[tid]);
}

__global__ void transport_fixed_source_kernel(curandState* states,
                                              unsigned int histories,
                                              MultiGroupMaterial material,
                                              FixedSource source,
                                              Vec3 min_bound,
                                              Vec3 max_bound,
                                              unsigned int max_collisions,
                                              float dx,
                                              float dy,
                                              float dz,
                                              float* flux_tally,
                                              unsigned long long* reaction_counts,
                                              float* track_lengths)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= histories)
    {
        return;
    }

    curandState state = states[tid];
    Vec3 position = make_vec3(curand_uniform(&state) * (max_bound.x - min_bound.x) + min_bound.x,
                              curand_uniform(&state) * (max_bound.y - min_bound.y) + min_bound.y,
                              curand_uniform(&state) * (max_bound.z - min_bound.z) + min_bound.z);
    Vec3 direction = sample_isotropic_direction(&state);
    int group = sample_group_from_chi(source.chi, &state);
    float weight = 1.0f;

    float track_length = 0.0f;
    bool alive = true;

    for (unsigned int collision = 0; collision < max_collisions && alive; ++collision)
    {
        const float sigma_t = material.total_xs(group);
        if (sigma_t <= 0.0f)
        {
            break;
        }

        const float free_path = sample_free_path(&state, sigma_t);
        const float boundary_distance = distance_to_boundary(position, direction, min_bound, max_bound);
        const float flight = fminf(free_path, boundary_distance);
        const Vec3 new_position = mad(position, direction, flight);
        const Vec3 midpoint = mad(position, direction, 0.5f * flight);

        if (inside_domain(midpoint, min_bound, max_bound))
        {
            const int cell = cell_index_from_position(midpoint, min_bound, dx, dy, dz);
            atomicAdd(&flux_tally[group * kNumCells + cell], weight * flight);
        }

        track_length += flight;
        position = new_position;

        if (boundary_distance <= free_path)
        {
            atomicAdd64(&reaction_counts[kGroups * kReactionPerGroup], 1ull);
            alive = false;
            break;
        }

        const float sigma_s_total = material.scatter_total(group);
        const float sigma_a = material.sigma_abs[group];
        const float sigma_f = material.sigma_f[group];

        float xi = curand_uniform(&state);
        const float p_s = sigma_s_total / sigma_t;
        const float p_a = sigma_a / sigma_t;

        if (xi < p_s)
        {
            const int base_index = group * kReactionPerGroup;
            atomicAdd64(&reaction_counts[base_index + 0], 1ull);
            const int scatter_group = sample_scatter_group(material, group, sigma_s_total, &state);
            group = scatter_group;
            direction = sample_isotropic_direction(&state);
            continue;
        }

        xi -= p_s;
        if (xi < p_a)
        {
            const int base_index = group * kReactionPerGroup;
            atomicAdd64(&reaction_counts[base_index + 1], 1ull);
            alive = false;
        }
        else
        {
            const int base_index = group * kReactionPerGroup;
            atomicAdd64(&reaction_counts[base_index + 2], 1ull);
            alive = false;
        }
    }

    track_lengths[tid] = track_length;
    states[tid] = state;
}

__global__ void transport_eigenvalue_kernel(curandState* states,
                                            unsigned int histories,
                                            MultiGroupMaterial material,
                                            Vec3 min_bound,
                                            Vec3 max_bound,
                                            unsigned int max_collisions,
                                            float dx,
                                            float dy,
                                            float dz,
                                            const ParticleSource* sources,
                                            ParticleSource* next_sources,
                                            unsigned int max_next,
                                            unsigned int* next_count,
                                            float* flux_tally,
                                            unsigned long long* reaction_counts,
                                            float* track_lengths,
                                            float* production_per_history)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= histories)
    {
        return;
    }

    curandState state = states[tid];
    ParticleSource particle = sources[tid];
    Vec3 position = particle.position;
    Vec3 direction = particle.direction;
    int group = particle.group;
    float weight = particle.weight;

    float track_length = 0.0f;
    float produced_weight = 0.0f;
    bool alive = true;

    for (unsigned int collision = 0; collision < max_collisions && alive; ++collision)
    {
        const float sigma_t = material.total_xs(group);
        if (sigma_t <= 0.0f)
        {
            break;
        }

        const float free_path = sample_free_path(&state, sigma_t);
        const float boundary_distance = distance_to_boundary(position, direction, min_bound, max_bound);
        const float flight = fminf(free_path, boundary_distance);
        const Vec3 new_position = mad(position, direction, flight);
        const Vec3 midpoint = mad(position, direction, 0.5f * flight);

        if (inside_domain(midpoint, min_bound, max_bound))
        {
            const int cell = cell_index_from_position(midpoint, min_bound, dx, dy, dz);
            atomicAdd(&flux_tally[group * kNumCells + cell], weight * flight);
        }

        track_length += flight;
        position = new_position;

        if (boundary_distance <= free_path)
        {
            atomicAdd64(&reaction_counts[kGroups * kReactionPerGroup], 1ull);
            alive = false;
            break;
        }

        const float sigma_s_total = material.scatter_total(group);
        const float sigma_a = material.sigma_abs[group];
        const float sigma_f = material.sigma_f[group];

        float xi = curand_uniform(&state);
        const float p_s = sigma_s_total / sigma_t;
        const float p_a = sigma_a / sigma_t;

        if (xi < p_s)
        {
            const int base_index = group * kReactionPerGroup;
            atomicAdd64(&reaction_counts[base_index + 0], 1ull);
            const int scatter_group = sample_scatter_group(material, group, sigma_s_total, &state);
            group = scatter_group;
            direction = sample_isotropic_direction(&state);
            continue;
        }

        xi -= p_s;
        if (xi < p_a)
        {
            const int base_index = group * kReactionPerGroup;
            atomicAdd64(&reaction_counts[base_index + 1], 1ull);
            alive = false;
        }
        else
        {
            const int base_index = group * kReactionPerGroup;
            atomicAdd64(&reaction_counts[base_index + 2], 1ull);
            produced_weight += weight * material.nu[group];

            ParticleSource offspring;
            offspring.position = position;
            offspring.direction = sample_isotropic_direction(&state);
            offspring.group = sample_group_from_chi(material.chi, &state);
            offspring.weight = weight * material.nu[group];

            const unsigned int index = atomicAdd(next_count, 1u);
            if (index < max_next)
            {
                next_sources[index] = offspring;
            }
            alive = false;
        }
    }

    track_lengths[tid] = track_length;
    production_per_history[tid] = produced_weight;
    states[tid] = state;
}

void run_fixed_source(unsigned int histories,
                      unsigned long long seed,
                      const MultiGroupMaterial& material,
                      const FixedSource& source,
                      const Vec3& min_bound,
                      const Vec3& max_bound,
                      unsigned int max_collisions)
{
    const float dx = (max_bound.x - min_bound.x) / static_cast<float>(kNx);
    const float dy = (max_bound.y - min_bound.y) / static_cast<float>(kNy);
    const float dz = (max_bound.z - min_bound.z) / static_cast<float>(kNz);

    const size_t state_bytes = sizeof(curandState) * histories;
    const size_t flux_bytes = sizeof(float) * kGroups * kNumCells;
    const size_t reaction_bytes = sizeof(unsigned long long) * kReactionCounts;
    const size_t track_bytes = sizeof(float) * histories;

    curandState* d_states = nullptr;
    float* d_flux = nullptr;
    unsigned long long* d_reactions = nullptr;
    float* d_tracks = nullptr;

    CUDA_CHECK(cudaMalloc(&d_states, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_flux, flux_bytes));
    CUDA_CHECK(cudaMalloc(&d_reactions, reaction_bytes));
    CUDA_CHECK(cudaMalloc(&d_tracks, track_bytes));

    CUDA_CHECK(cudaMemset(d_flux, 0, flux_bytes));
    CUDA_CHECK(cudaMemset(d_reactions, 0, reaction_bytes));

    const unsigned int threads_per_block = 256u;
    const unsigned int blocks = (histories + threads_per_block - 1u) / threads_per_block;

    initialize_rng<<<blocks, threads_per_block>>>(d_states, seed, 0ull, histories);
    CUDA_CHECK(cudaGetLastError());

    transport_fixed_source_kernel<<<blocks, threads_per_block>>>(d_states,
                                                                 histories,
                                                                 material,
                                                                 source,
                                                                 min_bound,
                                                                 max_bound,
                                                                 max_collisions,
                                                                 dx,
                                                                 dy,
                                                                 dz,
                                                                 d_flux,
                                                                 d_reactions,
                                                                 d_tracks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_flux(kGroups * kNumCells, 0.0f);
    std::vector<float> h_tracks(histories, 0.0f);
    std::array<unsigned long long, kReactionCounts> h_reactions{};

    CUDA_CHECK(cudaMemcpy(h_flux.data(), d_flux, flux_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tracks.data(), d_tracks, track_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_reactions.data(), d_reactions, reaction_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_flux));
    CUDA_CHECK(cudaFree(d_reactions));
    CUDA_CHECK(cudaFree(d_tracks));

    const double total_track = std::accumulate(h_tracks.begin(), h_tracks.end(), 0.0);
    const double mean_track = total_track / static_cast<double>(histories);
    const double volume = static_cast<double>(dx) * dy * dz;

    std::array<double, kGroups> group_flux_mean{};
    for (int g = 0; g < kGroups; ++g)
    {
        double sum = 0.0;
        for (int cell = 0; cell < kNumCells; ++cell)
        {
            sum += h_flux[g * kNumCells + cell];
        }
        group_flux_mean[g] = sum / (static_cast<double>(histories) * volume * kNumCells);
    }

    std::cout << "Fixed source Monte Carlo (3-D multi-group)" << '\n';
    std::cout << "=========================================" << '\n';
    std::cout << "Histories simulated : " << histories << '\n';
    std::cout << "Average track length: " << mean_track << " cm" << '\n';

    for (int g = 0; g < kGroups; ++g)
    {
        const int base_index = g * kReactionPerGroup;
        std::cout << "Group " << g << " scatter events   : " << h_reactions[base_index + 0] << '\n';
        std::cout << "Group " << g << " absorptions      : " << h_reactions[base_index + 1] << '\n';
        std::cout << "Group " << g << " fissions         : " << h_reactions[base_index + 2] << '\n';
        std::cout << "Group " << g << " mean flux (1/cm^2): " << group_flux_mean[g] << '\n';
    }
    std::cout << "Leakage events      : " << h_reactions[kGroups * kReactionPerGroup] << '\n';
    std::cout << '\n';

    const int center_ix = kNx / 2;
    const int center_iy = kNy / 2;
    std::cout << "Axial flux along z through domain center" << '\n';
    std::cout << "z (cm)";
    for (int g = 0; g < kGroups; ++g)
    {
        std::cout << std::setw(18) << ("Group " + std::to_string(g));
    }
    std::cout << '\n';

    for (int iz = 0; iz < kNz; ++iz)
    {
        const int cell_index = center_ix + center_iy * kNx + iz * kNx * kNy;
        const double z_center = (iz + 0.5) * dz;
        std::cout << std::setw(7) << std::setprecision(4) << z_center;
        for (int g = 0; g < kGroups; ++g)
        {
            const double flux = h_flux[g * kNumCells + cell_index] /
                                 (static_cast<double>(histories) * volume);
            std::cout << std::setw(18) << std::setprecision(6) << flux;
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

void run_eigenvalue(unsigned int histories,
                    unsigned long long seed,
                    unsigned int generations,
                    unsigned int inactive,
                    const MultiGroupMaterial& material,
                    const Vec3& min_bound,
                    const Vec3& max_bound,
                    unsigned int max_collisions)
{
    if (generations == 0)
    {
        return;
    }

    const float dx = (max_bound.x - min_bound.x) / static_cast<float>(kNx);
    const float dy = (max_bound.y - min_bound.y) / static_cast<float>(kNy);
    const float dz = (max_bound.z - min_bound.z) / static_cast<float>(kNz);

    const size_t state_bytes = sizeof(curandState) * histories;
    const size_t source_bytes = sizeof(ParticleSource) * histories;
    const size_t next_bytes = sizeof(ParticleSource) * histories * 8ull;
    const size_t flux_bytes = sizeof(float) * kGroups * kNumCells;
    const size_t reaction_bytes = sizeof(unsigned long long) * kReactionCounts;
    const size_t track_bytes = sizeof(float) * histories;
    const size_t production_bytes = sizeof(float) * histories;

    curandState* d_states = nullptr;
    ParticleSource* d_sources = nullptr;
    ParticleSource* d_next_sources = nullptr;
    unsigned int* d_next_count = nullptr;
    float* d_flux = nullptr;
    unsigned long long* d_reactions = nullptr;
    float* d_tracks = nullptr;
    float* d_production = nullptr;

    CUDA_CHECK(cudaMalloc(&d_states, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_sources, source_bytes));
    CUDA_CHECK(cudaMalloc(&d_next_sources, next_bytes));
    CUDA_CHECK(cudaMalloc(&d_next_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_flux, flux_bytes));
    CUDA_CHECK(cudaMalloc(&d_reactions, reaction_bytes));
    CUDA_CHECK(cudaMalloc(&d_tracks, track_bytes));
    CUDA_CHECK(cudaMalloc(&d_production, production_bytes));

    const unsigned int threads_per_block = 256u;
    const unsigned int blocks = (histories + threads_per_block - 1u) / threads_per_block;

    initialize_rng<<<blocks, threads_per_block>>>(d_states, seed, 0ull, histories);
    CUDA_CHECK(cudaGetLastError());

    std::vector<ParticleSource> h_sources(histories);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    for (unsigned int i = 0; i < histories; ++i)
    {
        ParticleSource src;
        src.position = make_vec3(uniform01(rng) * (max_bound.x - min_bound.x) + min_bound.x,
                                 uniform01(rng) * (max_bound.y - min_bound.y) + min_bound.y,
                                 uniform01(rng) * (max_bound.z - min_bound.z) + min_bound.z);
        const float mu = 2.0f * uniform01(rng) - 1.0f;
        const float phi = 2.0f * kPi * uniform01(rng);
        const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - mu * mu));
        src.direction = make_vec3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), mu);
        const float xi = uniform01(rng);
        src.group = (xi < material.chi[0]) ? 0 : 1;
        src.weight = 1.0f;
        h_sources[i] = src;
    }

    CUDA_CHECK(cudaMemcpy(d_sources, h_sources.data(), source_bytes, cudaMemcpyHostToDevice));

    std::vector<float> flux_accum(kGroups * kNumCells, 0.0f);
    std::array<unsigned long long, kReactionCounts> reaction_accum{};
    std::vector<double> keff_values;

    std::vector<float> h_flux(kGroups * kNumCells, 0.0f);
    std::vector<float> h_tracks(histories, 0.0f);
    std::vector<float> h_production(histories, 0.0f);

    for (unsigned int gen = 0; gen < generations; ++gen)
    {
        CUDA_CHECK(cudaMemset(d_flux, 0, flux_bytes));
        CUDA_CHECK(cudaMemset(d_reactions, 0, reaction_bytes));
        CUDA_CHECK(cudaMemset(d_tracks, 0, track_bytes));
        CUDA_CHECK(cudaMemset(d_production, 0, production_bytes));
        CUDA_CHECK(cudaMemset(d_next_count, 0, sizeof(unsigned int)));

        transport_eigenvalue_kernel<<<blocks, threads_per_block>>>(d_states,
                                                                   histories,
                                                                   material,
                                                                   min_bound,
                                                                   max_bound,
                                                                   max_collisions,
                                                                   dx,
                                                                   dy,
                                                                   dz,
                                                                   d_sources,
                                                                   d_next_sources,
                                                                   histories * 8u,
                                                                   d_next_count,
                                                                   d_flux,
                                                                   d_reactions,
                                                                   d_tracks,
                                                                   d_production);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_flux.data(), d_flux, flux_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tracks.data(), d_tracks, track_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_production.data(), d_production, production_bytes, cudaMemcpyDeviceToHost));

        std::array<unsigned long long, kReactionCounts> h_reactions{};
        CUDA_CHECK(cudaMemcpy(h_reactions.data(), d_reactions, reaction_bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < kReactionCounts; ++i)
        {
            reaction_accum[i] += h_reactions[i];
        }

        double production_sum = 0.0;
        for (float value : h_production)
        {
            production_sum += static_cast<double>(value);
        }

        const double total_weight_prev = static_cast<double>(histories);
        const double keff_gen = (total_weight_prev > 0.0) ? (production_sum / total_weight_prev) : 0.0;
        if (gen >= inactive)
        {
            for (size_t i = 0; i < flux_accum.size(); ++i)
            {
                flux_accum[i] += h_flux[i];
            }
            keff_values.push_back(keff_gen);
        }

        unsigned int next_count = 0u;
        CUDA_CHECK(cudaMemcpy(&next_count, d_next_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        next_count = std::min(next_count, histories * 8u);

        std::vector<ParticleSource> h_next(next_count);
        if (next_count > 0)
        {
            CUDA_CHECK(cudaMemcpy(h_next.data(), d_next_sources, next_count * sizeof(ParticleSource), cudaMemcpyDeviceToHost));
        }

        if (next_count == 0)
        {
            std::cerr << "Warning: eigenvalue iteration produced no offspring in generation " << gen << '\n';
            break;
        }

        std::vector<double> cumulative(next_count, 0.0);
        double total_weight = 0.0;
        for (unsigned int i = 0; i < next_count; ++i)
        {
            total_weight += static_cast<double>(h_next[i].weight);
            cumulative[i] = total_weight;
        }

        if (total_weight <= 0.0)
        {
            std::cerr << "Warning: zero total weight in generation " << gen << '\n';
            break;
        }

        std::uniform_real_distribution<double> uniform_weight(0.0, total_weight);
        for (unsigned int i = 0; i < histories; ++i)
        {
            const double xi = uniform_weight(rng);
            const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), xi);
            const size_t idx = static_cast<size_t>(std::distance(cumulative.begin(), it));
            ParticleSource selected = h_next[std::min(idx, static_cast<size_t>(next_count - 1))];
            selected.weight = 1.0f;
            h_sources[i] = selected;
        }

        CUDA_CHECK(cudaMemcpy(d_sources, h_sources.data(), source_bytes, cudaMemcpyHostToDevice));
    }

    const double volume = static_cast<double>(dx) * dy * dz;
    const unsigned int active_gens = (generations > inactive) ? (generations - inactive) : 0;

    std::cout << "Eigenvalue Monte Carlo (3-D multi-group)" << '\n';
    std::cout << "========================================" << '\n';
    std::cout << "Histories per generation: " << histories << '\n';
    std::cout << "Generations simulated   : " << generations << " (inactive: " << inactive << ")" << '\n';

    if (!keff_values.empty())
    {
        const double keff_mean = std::accumulate(keff_values.begin(), keff_values.end(), 0.0) /
                                 static_cast<double>(keff_values.size());
        double keff_sq = 0.0;
        for (double v : keff_values)
        {
            keff_sq += v * v;
        }
        const double variance = (keff_sq / keff_values.size()) - keff_mean * keff_mean;
        const double keff_std = (keff_values.size() > 1) ? std::sqrt(std::max(0.0, variance / (keff_values.size() - 1))) : 0.0;
        std::cout << std::setprecision(6) << "Estimated k-effective : " << keff_mean
                  << " +- " << keff_std << '\n';
    }

    if (active_gens > 0)
    {
        for (int g = 0; g < kGroups; ++g)
        {
            double sum = 0.0;
            for (int cell = 0; cell < kNumCells; ++cell)
            {
                sum += flux_accum[g * kNumCells + cell];
            }
            const double mean_flux = sum / (static_cast<double>(histories) * active_gens * volume * kNumCells);
            std::cout << "Group " << g << " mean flux (1/cm^2): " << mean_flux << '\n';
        }
    }

    for (int g = 0; g < kGroups; ++g)
    {
        const int base_index = g * kReactionPerGroup;
        std::cout << "Group " << g << " scatter events   : " << reaction_accum[base_index + 0] << '\n';
        std::cout << "Group " << g << " absorptions      : " << reaction_accum[base_index + 1] << '\n';
        std::cout << "Group " << g << " fissions         : " << reaction_accum[base_index + 2] << '\n';
    }
    std::cout << "Leakage events accumulated: " << reaction_accum[kGroups * kReactionPerGroup] << '\n';

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_sources));
    CUDA_CHECK(cudaFree(d_next_sources));
    CUDA_CHECK(cudaFree(d_next_count));
    CUDA_CHECK(cudaFree(d_flux));
    CUDA_CHECK(cudaFree(d_reactions));
    CUDA_CHECK(cudaFree(d_tracks));
    CUDA_CHECK(cudaFree(d_production));
}

} // namespace

int main(int argc, char** argv)
{
    const unsigned int histories = (argc > 1) ? static_cast<unsigned int>(std::stoul(argv[1])) : 50000u;
    const unsigned long long seed = (argc > 2) ? std::stoull(argv[2]) : 2024ull;
    const unsigned int generations = (argc > 3) ? static_cast<unsigned int>(std::stoul(argv[3])) : 8u;
    const unsigned int inactive = (argc > 4) ? static_cast<unsigned int>(std::stoul(argv[4])) : 2u;
    const unsigned int max_collisions = 256u;

    MultiGroupMaterial material{};
    material.sigma_abs[0] = 0.05f;
    material.sigma_abs[1] = 0.06f;
    material.sigma_f[0] = 0.08f;
    material.sigma_f[1] = 0.12f;
    material.nu[0] = 2.50f;
    material.nu[1] = 2.30f;
    material.chi[0] = 0.7f;
    material.chi[1] = 0.3f;
    material.scatter[0][0] = 0.20f;
    material.scatter[0][1] = 0.30f;
    material.scatter[1][0] = 0.02f;
    material.scatter[1][1] = 0.40f;

    FixedSource source{};
    source.chi[0] = 0.9f;
    source.chi[1] = 0.1f;

    const Vec3 min_bound = make_vec3(0.0f, 0.0f, 0.0f);
    const Vec3 max_bound = make_vec3(10.0f, 10.0f, 10.0f);

    try
    {
        run_fixed_source(histories, seed, material, source, min_bound, max_bound, max_collisions);
        run_eigenvalue(histories, seed + 1337ull, generations, inactive, material, min_bound, max_bound, max_collisions);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Simulation failed: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
