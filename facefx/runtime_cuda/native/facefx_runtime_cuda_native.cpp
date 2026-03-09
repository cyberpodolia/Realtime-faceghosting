#include <cmath>
#include <cstddef>
#include <algorithm>

namespace {

inline float clampf(float value, float lo, float hi) {
    return std::max(lo, std::min(hi, value));
}

inline float edge_function(
    float ax,
    float ay,
    float bx,
    float by,
    float px,
    float py
) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

inline int reflect101(int v, int limit) {
    if (limit <= 1) {
        return 0;
    }
    while (v < 0 || v >= limit) {
        if (v < 0) {
            v = -v;
        }
        if (v >= limit) {
            v = 2 * limit - v - 2;
        }
    }
    return v;
}

inline float bilinear_sample_u8(
    const unsigned char* img,
    int img_h,
    int img_w,
    int channels,
    float x,
    float y,
    int channel
) {
    const float sx = clampf(x, 0.0f, static_cast<float>(img_w - 1));
    const float sy = clampf(y, 0.0f, static_cast<float>(img_h - 1));
    const int x0 = reflect101(static_cast<int>(std::floor(sx)), img_w);
    const int y0 = reflect101(static_cast<int>(std::floor(sy)), img_h);
    const int x1 = reflect101(x0 + 1, img_w);
    const int y1 = reflect101(y0 + 1, img_h);
    const float tx = sx - std::floor(sx);
    const float ty = sy - std::floor(sy);

    const std::size_t row0 = static_cast<std::size_t>(y0) * static_cast<std::size_t>(img_w);
    const std::size_t row1 = static_cast<std::size_t>(y1) * static_cast<std::size_t>(img_w);
    const float p00 = static_cast<float>(img[(row0 + static_cast<std::size_t>(x0)) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(channel)]);
    const float p10 = static_cast<float>(img[(row0 + static_cast<std::size_t>(x1)) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(channel)]);
    const float p01 = static_cast<float>(img[(row1 + static_cast<std::size_t>(x0)) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(channel)]);
    const float p11 = static_cast<float>(img[(row1 + static_cast<std::size_t>(x1)) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(channel)]);

    const float top = p00 + (p10 - p00) * tx;
    const float bottom = p01 + (p11 - p01) * tx;
    return top + (bottom - top) * ty;
}

} // namespace

// C ABI so Python ctypes can call this function directly.
extern "C" __declspec(dllexport) int build_dense_remap_idw_f32(
    const float* src_xy,
    const float* dst_xy,
    int n_points,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    float power,
    float eps,
    float* out_map_x,
    float* out_map_y
) {
    if (src_xy == nullptr || dst_xy == nullptr || out_map_x == nullptr || out_map_y == nullptr) {
        return 1;
    }
    if (n_points < 3 || roi_w < 1 || roi_h < 1 || power <= 0.0f || eps <= 0.0f) {
        return 2;
    }

    const float inv_power = -0.5f * power;
    const bool power_is_two = std::fabs(power - 2.0f) < 1e-6f;
    const std::size_t n = static_cast<std::size_t>(n_points);
    const std::size_t width = static_cast<std::size_t>(roi_w);
    const std::size_t height = static_cast<std::size_t>(roi_h);

    #pragma omp parallel for
    for (int iy_i = 0; iy_i < roi_h; ++iy_i) {
        const std::size_t iy = static_cast<std::size_t>(iy_i);
        const float gy = static_cast<float>(roi_y) + static_cast<float>(iy);
        for (std::size_t ix = 0; ix < width; ++ix) {
            const float gx = static_cast<float>(roi_x) + static_cast<float>(ix);
            float sum_w = 0.0f;
            float sum_dx = 0.0f;
            float sum_dy = 0.0f;

            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t p = i * 2;
                const float dst_x = dst_xy[p];
                const float dst_y = dst_xy[p + 1];
                const float src_x = src_xy[p];
                const float src_y = src_xy[p + 1];

                const float ddx = gx - dst_x;
                const float ddy = gy - dst_y;
                const float dist2 = ddx * ddx + ddy * ddy + eps;
                const float w = power_is_two ? (1.0f / dist2) : std::pow(dist2, inv_power);

                sum_w += w;
                sum_dx += w * (src_x - dst_x);
                sum_dy += w * (src_y - dst_y);
            }

            const std::size_t out_index = iy * width + ix;
            if (sum_w <= 0.0f) {
                out_map_x[out_index] = gx;
                out_map_y[out_index] = gy;
            } else {
                out_map_x[out_index] = gx + (sum_dx / sum_w);
                out_map_y[out_index] = gy + (sum_dy / sum_w);
            }
        }
    }

    return 0;
}

extern "C" __declspec(dllexport) int warp_triangles_u8(
    const unsigned char* src_img,
    int src_h,
    int src_w,
    int channels,
    unsigned char* dst_img,
    int dst_h,
    int dst_w,
    const float* src_points,
    const float* dst_points,
    int n_points,
    const int* simplices,
    int n_tris,
    float* dst_mask
) {
    if (
        src_img == nullptr || dst_img == nullptr || src_points == nullptr || dst_points == nullptr ||
        simplices == nullptr || dst_mask == nullptr
    ) {
        return 1;
    }
    if (
        src_h < 1 || src_w < 1 || dst_h < 1 || dst_w < 1 ||
        (channels != 1 && channels != 3) || n_points < 3 || n_tris < 1
    ) {
        return 2;
    }

    const std::size_t dst_pixels = static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w);
    std::fill(dst_img, dst_img + dst_pixels * static_cast<std::size_t>(channels), static_cast<unsigned char>(0));
    std::fill(dst_mask, dst_mask + dst_pixels, 0.0f);

    for (int tri_i = 0; tri_i < n_tris; ++tri_i) {
        const int ia = simplices[tri_i * 3 + 0];
        const int ib = simplices[tri_i * 3 + 1];
        const int ic = simplices[tri_i * 3 + 2];
        if (ia < 0 || ib < 0 || ic < 0 || ia >= n_points || ib >= n_points || ic >= n_points) {
            continue;
        }

        const float dax = dst_points[ia * 2 + 0];
        const float day = dst_points[ia * 2 + 1];
        const float dbx = dst_points[ib * 2 + 0];
        const float dby = dst_points[ib * 2 + 1];
        const float dcx = dst_points[ic * 2 + 0];
        const float dcy = dst_points[ic * 2 + 1];

        const float sax = src_points[ia * 2 + 0];
        const float say = src_points[ia * 2 + 1];
        const float sbx = src_points[ib * 2 + 0];
        const float sby = src_points[ib * 2 + 1];
        const float scx = src_points[ic * 2 + 0];
        const float scy = src_points[ic * 2 + 1];

        const float area = edge_function(dax, day, dbx, dby, dcx, dcy);
        if (std::fabs(area) < 1e-6f) {
            continue;
        }

        const float min_x = std::min(dax, std::min(dbx, dcx));
        const float max_x = std::max(dax, std::max(dbx, dcx));
        const float min_y = std::min(day, std::min(dby, dcy));
        const float max_y = std::max(day, std::max(dby, dcy));

        const int x0 = std::max(0, static_cast<int>(std::floor(min_x)));
        const int x1 = std::min(dst_w - 1, static_cast<int>(std::ceil(max_x)));
        const int y0 = std::max(0, static_cast<int>(std::floor(min_y)));
        const int y1 = std::min(dst_h - 1, static_cast<int>(std::ceil(max_y)));
        if (x0 > x1 || y0 > y1) {
            continue;
        }

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const float px = static_cast<float>(x) + 0.5f;
                const float py = static_cast<float>(y) + 0.5f;
                const float w0 = edge_function(dbx, dby, dcx, dcy, px, py) / area;
                const float w1 = edge_function(dcx, dcy, dax, day, px, py) / area;
                const float w2 = 1.0f - w0 - w1;
                if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) {
                    continue;
                }

                const float src_x = sax * w0 + sbx * w1 + scx * w2;
                const float src_y = say * w0 + sby * w1 + scy * w2;
                const std::size_t dst_idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) + static_cast<std::size_t>(x);
                for (int c = 0; c < channels; ++c) {
                    const float sample = bilinear_sample_u8(src_img, src_h, src_w, channels, src_x, src_y, c);
                    dst_img[dst_idx * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] =
                        static_cast<unsigned char>(clampf(std::round(sample), 0.0f, 255.0f));
                }
                dst_mask[dst_idx] = 1.0f;
            }
        }
    }

    return 0;
}
