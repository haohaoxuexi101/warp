#include <cuda.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "check_cuda.h"

namespace
{
//! number of spatial bins used for the track-length flux tally
constexpr int kFluxBins = 64;
//! small value to prevent taking the logarithm of zero
constexpr float kEpsilon = 1.0e-7f;

struct Material
{
    float scatter_xs;   ///< macroscopic scattering cross section (1/cm)
    float absorption_xs;///< macroscopic absorption cross section (1/cm)
    float fission_xs;   ///< macroscopic fission cross section (1/cm)
    float nu;           ///< average neutrons produced per fission

    __host__ __device__ float total_xs() const
    {
        return scatter_xs + absorption_xs + fission_xs;
    }
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

__device__ inline float sample_isotropic_mu(curandState* state)
{
    const float xi = curand_uniform(state);
    return 2.0f * xi - 1.0f;
}

__device__ inline float sample_free_path(curandState* state, float sigma_t)
{
    float xi = curand_uniform(state);
    xi = fmaxf(xi, kEpsilon);
    return -logf(xi) / sigma_t;
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

__global__ void transport_kernel(curandState* states,
                                 unsigned int histories,
                                 Material material,
                                 float slab_thickness,
                                 unsigned int max_collisions,
                                 float bin_width,
                                 float* flux_tally,
                                 unsigned long long* reaction_counts,
                                 float* track_lengths)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= histories)
    {
        return;
    }

    curandState local_state = states[tid];
    float position = 0.5f * slab_thickness;
    float direction = sample_isotropic_mu(&local_state);
    float track_length = 0.0f;

    unsigned int scatter_events = 0;
    unsigned int absorption_events = 0;
    unsigned int fission_events = 0;
    bool leaked = false;
    bool alive = true;

    const float total_xs = material.total_xs();

    for (unsigned int collision = 0; collision < max_collisions && alive; ++collision)
    {
        float mu = direction;
        if (fabsf(mu) < 1.0e-6f)
        {
            mu = (mu >= 0.0f) ? 1.0e-6f : -1.0e-6f;
        }

        const float free_path = sample_free_path(&local_state, total_xs);
        float distance_to_boundary = (mu > 0.0f) ? (slab_thickness - position) / mu
                                                 : -position / mu;
        distance_to_boundary = fmaxf(distance_to_boundary, 0.0f);

        float flight_distance = free_path;
        bool crossed_boundary = false;
        if (flight_distance > distance_to_boundary)
        {
            flight_distance = distance_to_boundary;
            crossed_boundary = true;
        }

        const float new_position = position + mu * flight_distance;
        const float midpoint = 0.5f * (position + new_position);
        int bin = static_cast<int>(floorf(midpoint / bin_width));
        bin = max(0, min(bin, kFluxBins - 1));
        atomicAdd(&flux_tally[bin], flight_distance);

        track_length += flight_distance;
        position = new_position;

        if (crossed_boundary)
        {
            leaked = true;
            alive = false;
            break;
        }

        const float xi_rxn = curand_uniform(&local_state);
        const float scatter_prob = material.scatter_xs / total_xs;
        const float absorb_prob = (material.scatter_xs + material.absorption_xs) / total_xs;

        if (xi_rxn < scatter_prob)
        {
            ++scatter_events;
            direction = sample_isotropic_mu(&local_state);
        }
        else if (xi_rxn < absorb_prob)
        {
            ++absorption_events;
            alive = false;
        }
        else
        {
            ++fission_events;
            alive = false;
        }
    }

    track_lengths[tid] = track_length;
    states[tid] = local_state;

    if (scatter_events > 0)
    {
        atomicAdd64(&reaction_counts[0], static_cast<unsigned long long>(scatter_events));
    }
    if (absorption_events > 0)
    {
        atomicAdd64(&reaction_counts[1], static_cast<unsigned long long>(absorption_events));
    }
    if (fission_events > 0)
    {
        atomicAdd64(&reaction_counts[2], static_cast<unsigned long long>(fission_events));
    }
    if (leaked)
    {
        atomicAdd64(&reaction_counts[3], 1ull);
    }
}

} // namespace

int main(int argc, char** argv)
{
    const unsigned int histories = (argc > 1) ? static_cast<unsigned int>(std::stoul(argv[1])) : 1000000u;
    const unsigned long long seed = (argc > 2) ? std::stoull(argv[2]) : 1337ull;
    const unsigned int max_collisions = 1024u;
    const float slab_thickness = 10.0f; // cm
    const float bin_width = slab_thickness / static_cast<float>(kFluxBins);

    Material material;
    material.scatter_xs = 0.45f;   // 1/cm
    material.absorption_xs = 0.12f; // 1/cm
    material.fission_xs = 0.03f;    // 1/cm
    material.nu = 2.43f;

    const size_t state_bytes = sizeof(curandState) * histories;
    const size_t tally_bytes = sizeof(float) * kFluxBins;
    const size_t reaction_bytes = sizeof(unsigned long long) * 4ull;
    const size_t track_bytes = sizeof(float) * histories;

    curandState* d_states = nullptr;
    float* d_flux_tally = nullptr;
    unsigned long long* d_reaction_counts = nullptr;
    float* d_track_lengths = nullptr;

    check_cuda(cudaMalloc(&d_states, state_bytes));
    check_cuda(cudaMalloc(&d_flux_tally, tally_bytes));
    check_cuda(cudaMalloc(&d_reaction_counts, reaction_bytes));
    check_cuda(cudaMalloc(&d_track_lengths, track_bytes));

    check_cuda(cudaMemset(d_flux_tally, 0, tally_bytes));
    check_cuda(cudaMemset(d_reaction_counts, 0, reaction_bytes));

    const unsigned int threads_per_block = 256u;
    const unsigned int blocks = (histories + threads_per_block - 1u) / threads_per_block;

    initialize_rng<<<blocks, threads_per_block>>>(d_states, seed, 0ull, histories);
    check_cuda(cudaGetLastError());

    transport_kernel<<<blocks, threads_per_block>>>(d_states,
                                                    histories,
                                                    material,
                                                    slab_thickness,
                                                    max_collisions,
                                                    bin_width,
                                                    d_flux_tally,
                                                    d_reaction_counts,
                                                    d_track_lengths);
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    std::vector<float> h_flux_tally(kFluxBins, 0.0f);
    std::vector<float> h_track_lengths(histories, 0.0f);
    unsigned long long h_reaction_counts[4] = {0ull, 0ull, 0ull, 0ull};

    check_cuda(cudaMemcpy(h_flux_tally.data(), d_flux_tally, tally_bytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(h_track_lengths.data(), d_track_lengths, track_bytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(h_reaction_counts, d_reaction_counts, reaction_bytes, cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_states));
    check_cuda(cudaFree(d_flux_tally));
    check_cuda(cudaFree(d_reaction_counts));
    check_cuda(cudaFree(d_track_lengths));

    const double total_track_length = std::accumulate(h_track_lengths.begin(), h_track_lengths.end(), 0.0);
    const double mean_track_length = total_track_length / static_cast<double>(histories);

    const unsigned long long scatter_count = h_reaction_counts[0];
    const unsigned long long absorption_count = h_reaction_counts[1];
    const unsigned long long fission_count = h_reaction_counts[2];
    const unsigned long long leakage_count = h_reaction_counts[3];
    const unsigned long long collision_count = scatter_count + absorption_count + fission_count;

    const double leakage_fraction = static_cast<double>(leakage_count) / static_cast<double>(histories);
    const double absorption_fraction = static_cast<double>(absorption_count) / static_cast<double>(histories);
    const double fission_fraction = static_cast<double>(fission_count) / static_cast<double>(histories);

    std::cout << "CUDA Monte Carlo slab transport" << '\n';
    std::cout << "================================" << '\n';
    std::cout << "Histories simulated : " << histories << '\n';
    std::cout << "Average track length: " << mean_track_length << " cm" << '\n';
    std::cout << "Collisions sampled  : " << collision_count << '\n';
    std::cout << " Scatter events     : " << scatter_count << '\n';
    std::cout << " Absorptions        : " << absorption_count << '\n';
    std::cout << " Fissions           : " << fission_count << '\n';
    std::cout << " Leakage            : " << leakage_count << " (fraction = "
              << leakage_fraction << ")" << '\n';
    std::cout << " Absorption fraction: " << absorption_fraction << '\n';
    std::cout << " Fission fraction   : " << fission_fraction << '\n';

    if (absorption_count > 0)
    {
        const double kinf = (material.nu * static_cast<double>(fission_count)) /
                             static_cast<double>(absorption_count);
        std::cout << " Estimated k-infinity: " << kinf << '\n';
    }

    std::cout << '\n';
    std::cout << "Axial flux tally (track length per bin)" << '\n';
    std::cout << "Bin center (cm)" << std::setw(18) << "Track length (cm)" << '\n';

    for (int bin = 0; bin < kFluxBins; ++bin)
    {
        const double center = (bin + 0.5) * bin_width;
        std::cout << std::setw(12) << std::setprecision(4) << center
                  << std::setw(18) << std::setprecision(6) << h_flux_tally[bin]
                  << '\n';
    }

    return 0;
}
