#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <exception>
#include <cstdlib>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t cudaCheckStatus = (call);                                                    \
        if (cudaSuccess != cudaCheckStatus) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(cudaCheckStatus), \
                         __FILE__, __LINE__);                                                   \
            std::exit(EXIT_FAILURE);                                                             \
        }                                                                                        \
    } while (0)

struct Ray {
    float3 origin;
    float3 direction;
};

struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    int material;
};

struct Aabb {
    float3 min;
    float3 max;
};

struct BvhNode {
    float3 boundsMin;
    float3 boundsMax;
    int leftChild;
    int rightChild;
    int firstPrim;
    int primCount;
};

__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &v) {
    return make_float3(-v.x, -v.y, -v.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3 &a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline float3 operator*(float a, const float3 &b) {
    return b * a;
}

__host__ __device__ inline float3 operator/(const float3 &a, float b) {
    const float inv = 1.0f / b;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

__host__ __device__ inline float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(const float3 &a, const float3 &b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float3 normalize(const float3 &v) {
    const float len = sqrtf(dot(v, v));
    if (len == 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return v / len;
}

__host__ __device__ inline float3 min_vec(const float3 &a, const float3 &b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ inline float3 max_vec(const float3 &a, const float3 &b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ inline float3 lerp(const float3 &a, const float3 &b, float t) {
    return a * (1.0f - t) + b * t;
}

__host__ __device__ inline float component(const float3 &v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

__host__ __device__ inline float3 triangle_normal(const Triangle &tri) {
    return normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
}

__device__ bool intersect_aabb(const Ray &ray, const float3 &bmin, const float3 &bmax, float tMax) {
    float t0 = 0.0f;
    float t1 = tMax;
    for (int axis = 0; axis < 3; ++axis) {
        const float invD = 1.0f / component(ray.direction, axis);
        float tNear = (component(bmin, axis) - component(ray.origin, axis)) * invD;
        float tFar = (component(bmax, axis) - component(ray.origin, axis)) * invD;
        if (tNear > tFar) {
            const float tmp = tNear;
            tNear = tFar;
            tFar = tmp;
        }
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1) {
            return false;
        }
    }
    return true;
}

__device__ bool intersect_triangle(const Ray &ray, const Triangle &tri, float &t, float &u, float &v) {
    const float3 edge1 = tri.v1 - tri.v0;
    const float3 edge2 = tri.v2 - tri.v0;
    const float3 pvec = cross(ray.direction, edge2);
    const float det = dot(edge1, pvec);
    if (fabsf(det) < 1e-8f) {
        return false;
    }
    const float invDet = 1.0f / det;
    const float3 tvec = ray.origin - tri.v0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const float3 qvec = cross(tvec, edge1);
    v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }
    t = dot(edge2, qvec) * invDet;
    return t > 1e-5f;
}

__device__ float3 shade_material(int material, const float3 &normal) {
    const float3 baseColors[] = {
        make_float3(0.8f, 0.4f, 0.3f),
        make_float3(0.3f, 0.6f, 0.9f),
        make_float3(0.7f, 0.8f, 0.4f),
        make_float3(0.9f, 0.9f, 0.9f)};
    const int index = material % 4;
    const float3 lightDir = normalize(make_float3(-0.4f, -1.0f, -0.3f));
    const float diffuse = fmaxf(0.1f, dot(normal, -lightDir));
    return baseColors[index] * diffuse;
}

__device__ float3 trace_scene(const Ray &ray, const BvhNode *nodes, const Triangle *tris,
                              const int *primIndices) {
    constexpr int kMaxStack = 64;
    int stack[kMaxStack];
    int stackSize = 0;
    int current = 0;
    float closest = 1e30f;
    int bestPrim = -1;
    float3 bestNormal = make_float3(0.0f, 0.0f, 0.0f);
    int bestMaterial = 0;

    while (true) {
        const BvhNode &node = nodes[current];
        if (intersect_aabb(ray, node.boundsMin, node.boundsMax, closest)) {
            if (node.primCount > 0) {
                for (int i = 0; i < node.primCount; ++i) {
                    const int primIdx = primIndices[node.firstPrim + i];
                    const Triangle &tri = tris[primIdx];
                    float t, u, v;
                    if (intersect_triangle(ray, tri, t, u, v) && t < closest) {
                        closest = t;
                        bestPrim = primIdx;
                        bestNormal = triangle_normal(tri);
                        bestMaterial = tri.material;
                    }
                }
                if (stackSize == 0) {
                    break;
                }
                current = stack[--stackSize];
            } else {
                const int left = node.leftChild;
                const int right = node.rightChild;
                const bool hasLeft = left >= 0;
                const bool hasRight = right >= 0;
                if (hasLeft && hasRight) {
                    const float3 leftMin = nodes[left].boundsMin;
                    const float3 leftMax = nodes[left].boundsMax;
                    const float3 rightMin = nodes[right].boundsMin;
                    const float3 rightMax = nodes[right].boundsMax;
                    const bool hitLeft = intersect_aabb(ray, leftMin, leftMax, closest);
                    const bool hitRight = intersect_aabb(ray, rightMin, rightMax, closest);
                    if (hitLeft && hitRight) {
                        // Visit the nearer child first
                        const float3 leftCenter = (leftMin + leftMax) * 0.5f;
                        const float3 rightCenter = (rightMin + rightMax) * 0.5f;
                        const float leftDist = dot(leftCenter - ray.origin, ray.direction);
                        const float rightDist = dot(rightCenter - ray.origin, ray.direction);
                        if (leftDist < rightDist) {
                            stack[stackSize++] = right;
                            current = left;
                        } else {
                            stack[stackSize++] = left;
                            current = right;
                        }
                    } else if (hitLeft) {
                        current = left;
                    } else if (hitRight) {
                        current = right;
                    } else {
                        if (stackSize == 0) {
                            break;
                        }
                        current = stack[--stackSize];
                    }
                } else if (hasLeft) {
                    current = left;
                } else if (hasRight) {
                    current = right;
                } else {
                    if (stackSize == 0) {
                        break;
                    }
                    current = stack[--stackSize];
                }
            }
        } else {
            if (stackSize == 0) {
                break;
            }
            current = stack[--stackSize];
        }
    }

    if (bestPrim >= 0) {
        return shade_material(bestMaterial, bestNormal);
    }

    const float3 skyTop = make_float3(0.6f, 0.75f, 1.0f);
    const float3 skyBottom = make_float3(0.2f, 0.3f, 0.4f);
    const float t = 0.5f * (ray.direction.y + 1.0f);
    return lerp(skyBottom, skyTop, t);
}

__global__ void render_kernel(const BvhNode *nodes, const Triangle *tris,
                              const int *primIndices, float3 camPos, float3 camForward,
                              float3 camRight, float3 camUp, float fovY, int width, int height,
                              float3 *image) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float scale = tanf(0.5f * fovY);
    const float px = (2.0f * ((x + 0.5f) / static_cast<float>(width)) - 1.0f) * aspect * scale;
    const float py = (1.0f - 2.0f * ((y + 0.5f) / static_cast<float>(height))) * scale;

    float3 dir = camForward + px * camRight + py * camUp;
    dir = normalize(dir);
    const Ray ray{camPos, dir};
    image[y * width + x] = trace_scene(ray, nodes, tris, primIndices);
}

float3 make_host_float3(float x, float y, float z) {
    float3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

Triangle make_triangle(const float3 &a, const float3 &b, const float3 &c, int material) {
    Triangle t{};
    t.v0 = a;
    t.v1 = b;
    t.v2 = c;
    t.material = material;
    return t;
}

void append_box(std::vector<Triangle> &tris, const float3 &minPt, const float3 &maxPt, int material) {
    const float3 v000 = minPt;
    const float3 v001 = make_host_float3(minPt.x, minPt.y, maxPt.z);
    const float3 v010 = make_host_float3(minPt.x, maxPt.y, minPt.z);
    const float3 v011 = make_host_float3(minPt.x, maxPt.y, maxPt.z);
    const float3 v100 = make_host_float3(maxPt.x, minPt.y, minPt.z);
    const float3 v101 = make_host_float3(maxPt.x, minPt.y, maxPt.z);
    const float3 v110 = make_host_float3(maxPt.x, maxPt.y, minPt.z);
    const float3 v111 = maxPt;

    // +X face
    tris.push_back(make_triangle(v100, v101, v111, material));
    tris.push_back(make_triangle(v100, v111, v110, material));
    // -X face
    tris.push_back(make_triangle(v000, v011, v001, material));
    tris.push_back(make_triangle(v000, v010, v011, material));
    // +Y face
    tris.push_back(make_triangle(v010, v111, v110, material));
    tris.push_back(make_triangle(v010, v011, v111, material));
    // -Y face
    tris.push_back(make_triangle(v000, v101, v100, material));
    tris.push_back(make_triangle(v000, v001, v101, material));
    // +Z face
    tris.push_back(make_triangle(v001, v111, v101, material));
    tris.push_back(make_triangle(v001, v011, v111, material));
    // -Z face
    tris.push_back(make_triangle(v000, v100, v110, material));
    tris.push_back(make_triangle(v000, v110, v010, material));
}

std::vector<Triangle> create_scene_geometry() {
    std::vector<Triangle> tris;
    tris.reserve(256);

    // Floor and ceiling slabs
    append_box(tris, make_host_float3(-10.0f, -0.5f, -10.0f), make_host_float3(10.0f, 0.0f, 10.0f), 0);
    append_box(tris, make_host_float3(-10.0f, 5.0f, -10.0f), make_host_float3(10.0f, 5.5f, 10.0f), 0);

    // Four surrounding walls
    append_box(tris, make_host_float3(-10.5f, 0.0f, -10.5f), make_host_float3(-10.0f, 5.0f, 10.5f), 1);
    append_box(tris, make_host_float3(10.0f, 0.0f, -10.5f), make_host_float3(10.5f, 5.0f, 10.5f), 1);
    append_box(tris, make_host_float3(-10.5f, 0.0f, -10.5f), make_host_float3(10.5f, 5.0f, -10.0f), 2);
    append_box(tris, make_host_float3(-10.5f, 0.0f, 10.0f), make_host_float3(10.5f, 5.0f, 10.5f), 2);

    // Central moderating block
    append_box(tris, make_host_float3(-1.5f, 0.0f, -1.5f), make_host_float3(1.5f, 3.0f, 1.5f), 3);

    // Detector column
    append_box(tris, make_host_float3(-4.0f, 0.0f, -0.5f), make_host_float3(-3.0f, 4.0f, 0.5f), 1);

    // Shielding slab with narrow channel
    append_box(tris, make_host_float3(3.0f, 0.0f, -2.0f), make_host_float3(6.0f, 4.0f, 2.0f), 2);
    append_box(tris, make_host_float3(3.5f, 1.0f, -0.4f), make_host_float3(6.0f, 3.0f, 0.4f), 0);

    return tris;
}

Aabb compute_bounds(const std::vector<Triangle> &tris, const std::vector<int> &indices, int start,
                    int count) {
    Aabb bounds;
    bounds.min = make_host_float3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max());
    bounds.max = make_host_float3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                                  -std::numeric_limits<float>::max());
    for (int i = 0; i < count; ++i) {
        const Triangle &tri = tris[indices[start + i]];
        bounds.min = min_vec(bounds.min, tri.v0);
        bounds.min = min_vec(bounds.min, tri.v1);
        bounds.min = min_vec(bounds.min, tri.v2);
        bounds.max = max_vec(bounds.max, tri.v0);
        bounds.max = max_vec(bounds.max, tri.v1);
        bounds.max = max_vec(bounds.max, tri.v2);
    }
    return bounds;
}

Aabb compute_centroid_bounds(const std::vector<Triangle> &tris, const std::vector<int> &indices,
                             int start, int count) {
    Aabb bounds;
    bounds.min = make_host_float3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max());
    bounds.max = make_host_float3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                                  -std::numeric_limits<float>::max());
    for (int i = 0; i < count; ++i) {
        const Triangle &tri = tris[indices[start + i]];
        const float3 centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
        bounds.min = min_vec(bounds.min, centroid);
        bounds.max = max_vec(bounds.max, centroid);
    }
    return bounds;
}

int build_bvh_recursive(const std::vector<Triangle> &tris, std::vector<BvhNode> &nodes,
                        std::vector<int> &primIndices, int start, int count) {
    const int nodeIndex = static_cast<int>(nodes.size());
    nodes.push_back({});

    const Aabb bounds = compute_bounds(tris, primIndices, start, count);
    nodes[nodeIndex].boundsMin = bounds.min;
    nodes[nodeIndex].boundsMax = bounds.max;

    if (count <= 4) {
        nodes[nodeIndex].firstPrim = start;
        nodes[nodeIndex].primCount = count;
        nodes[nodeIndex].leftChild = -1;
        nodes[nodeIndex].rightChild = -1;
        return nodeIndex;
    }

    const Aabb centroidBounds = compute_centroid_bounds(tris, primIndices, start, count);
    const float3 extents = centroidBounds.max - centroidBounds.min;
    int axis = 0;
    float maxExtent = extents.x;
    if (extents.y > maxExtent) {
        axis = 1;
        maxExtent = extents.y;
    }
    if (extents.z > maxExtent) {
        axis = 2;
        maxExtent = extents.z;
    }

    if (maxExtent < 1e-4f) {
        nodes[nodeIndex].firstPrim = start;
        nodes[nodeIndex].primCount = count;
        nodes[nodeIndex].leftChild = -1;
        nodes[nodeIndex].rightChild = -1;
        return nodeIndex;
    }

    const int mid = start + count / 2;
    auto comparator = [&](int a, int b) {
        const float3 ca = (tris[a].v0 + tris[a].v1 + tris[a].v2) / 3.0f;
        const float3 cb = (tris[b].v0 + tris[b].v1 + tris[b].v2) / 3.0f;
        return component(ca, axis) < component(cb, axis);
    };
    std::nth_element(primIndices.begin() + start, primIndices.begin() + mid,
                     primIndices.begin() + start + count, comparator);

    const int leftChild = build_bvh_recursive(tris, nodes, primIndices, start, mid - start);
    const int rightChild = build_bvh_recursive(tris, nodes, primIndices, mid, count - (mid - start));

    nodes[nodeIndex].leftChild = leftChild;
    nodes[nodeIndex].rightChild = rightChild;
    nodes[nodeIndex].firstPrim = -1;
    nodes[nodeIndex].primCount = 0;
    return nodeIndex;
}

std::vector<BvhNode> build_bvh(const std::vector<Triangle> &tris, std::vector<int> &primIndices) {
    std::vector<BvhNode> nodes;
    nodes.reserve(tris.size() * 2);
    build_bvh_recursive(tris, nodes, primIndices, 0, static_cast<int>(primIndices.size()));
    return nodes;
}

void write_ppm(const std::string &filename, int width, int height, const std::vector<float3> &image) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        const float3 color = image[i];
        const unsigned char r = static_cast<unsigned char>(255.99f * fminf(fmaxf(color.x, 0.0f), 1.0f));
        const unsigned char g = static_cast<unsigned char>(255.99f * fminf(fmaxf(color.y, 0.0f), 1.0f));
        const unsigned char b = static_cast<unsigned char>(255.99f * fminf(fmaxf(color.z, 0.0f), 1.0f));
        file.put(static_cast<char>(r));
        file.put(static_cast<char>(g));
        file.put(static_cast<char>(b));
    }
}

int main() {
    try {
        const int width = 640;
        const int height = 360;
        const float fovY = 45.0f * static_cast<float>(M_PI) / 180.0f;

        auto triangles = create_scene_geometry();
        if (triangles.empty()) {
            std::cerr << "No geometry generated for the scene." << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<int> primIndices(triangles.size());
        for (size_t i = 0; i < primIndices.size(); ++i) {
            primIndices[i] = static_cast<int>(i);
        }
        auto nodes = build_bvh(triangles, primIndices);

        BvhNode *d_nodes = nullptr;
        Triangle *d_tris = nullptr;
        int *d_primIndices = nullptr;
        float3 *d_image = nullptr;

        CUDA_CHECK(cudaMalloc(&d_nodes, nodes.size() * sizeof(BvhNode)));
        CUDA_CHECK(cudaMalloc(&d_tris, triangles.size() * sizeof(Triangle)));
        CUDA_CHECK(cudaMalloc(&d_primIndices, primIndices.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_image, width * height * sizeof(float3)));

        CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tris, triangles.data(), triangles.size() * sizeof(Triangle),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_primIndices, primIndices.data(), primIndices.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

        const float3 camPos = make_host_float3(0.0f, 2.5f, 12.0f);
        const float3 target = make_host_float3(0.0f, 1.5f, 0.0f);
        const float3 up = make_host_float3(0.0f, 1.0f, 0.0f);
        float3 camForward = normalize(target - camPos);
        float3 camRight = normalize(cross(camForward, up));
        float3 camUp = normalize(cross(camRight, camForward));

        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);
        render_kernel<<<gridDim, blockDim>>>(d_nodes, d_tris, d_primIndices, camPos, camForward,
                                             camRight, camUp, fovY, width, height, d_image);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float3> hostImage(width * height);
        CUDA_CHECK(cudaMemcpy(hostImage.data(), d_image, width * height * sizeof(float3),
                              cudaMemcpyDeviceToHost));

        write_ppm("ray_traced_scene.ppm", width, height, hostImage);
        std::cout << "Rendered image written to ray_traced_scene.ppm" << std::endl;

        CUDA_CHECK(cudaFree(d_nodes));
        CUDA_CHECK(cudaFree(d_tris));
        CUDA_CHECK(cudaFree(d_primIndices));
        CUDA_CHECK(cudaFree(d_image));
    } catch (const std::exception &ex) {
        std::cerr << "Unhandled exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
