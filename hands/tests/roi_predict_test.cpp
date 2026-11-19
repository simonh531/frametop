// Unit tests for velocity-based active ROI prediction (track/roi_predict.h):
// no ncnn, no models, no recordings. Run via `make check` or `just test-c`.
#include "../track/roi_predict.h"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;
#define CHECK(c)                                                                           \
    do {                                                                                   \
        if (!(c)) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c), ++failures;     \
    } while (0)

struct TestRoi {
    V2 center{};
    double size = 0, rotation = 0;
};

// Simple pinhole camera at origin looking along -z (OpenVR head frame: +x right, +y up, -z forward).
struct TestCamera {
    int width = 1000, height = 1000;
    double fx = 500, fy = 500, cx = 500, cy = 500;

    V2 project(V3 p, double *depth) const {
        const double z = -p[2];
        if (depth) *depth = z;
        if (z <= 1e-6) return {cx, cy};
        return {cx + fx * p[0] / z, cy - fy * p[1] / z};
    }
};

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) <= eps; }

}  // namespace

int main() {
    const TestCamera cam;
    const TestRoi base{{500.0, 500.0}, 200.0, 0.42};
    const V3 palm{0.0, 0.0, -0.5};   // 0.5 m straight ahead -> (500, 500)

    // 1. Gain 0 (off) leaves ROI untouched.
    {
        RoiPredictParams p;
        p.gain = 0.0;
        const TestRoi r = predict_roi(cam, base, palm, {1.0, 0.0, 0.0}, 0.033, p);
        CHECK(near(r.center[0], base.center[0]) && near(r.center[1], base.center[1]));
        CHECK(near(r.size, base.size) && near(r.rotation, base.rotation));
    }

    // 2. Non-positive dt or size leaves ROI untouched.
    {
        RoiPredictParams p;
        p.gain = 0.7;
        const TestRoi r1 = predict_roi(cam, base, palm, {1.0, 0.0, 0.0}, 0.0, p);
        const TestRoi r2 = predict_roi(cam, base, palm, {1.0, 0.0, 0.0}, -0.05, p);
        CHECK(near(r1.center[0], base.center[0]) && near(r2.center[0], base.center[0]));
        TestRoi zero_sz = base;
        zero_sz.size = 0;
        const TestRoi r3 = predict_roi(cam, zero_sz, palm, {1.0, 0.0, 0.0}, 0.033, p);
        CHECK(near(r3.center[0], zero_sz.center[0]));
    }

    // 3. Speed below min_speed (noise deadzone) leaves ROI untouched.
    {
        RoiPredictParams p;
        p.gain = 0.7;
        p.min_speed = 0.05;
        const TestRoi r = predict_roi(cam, base, palm, {0.03, 0.02, 0.0}, 0.033, p);
        CHECK(near(r.center[0], base.center[0]) && near(r.center[1], base.center[1]));
    }

    // 4. Lateral motion shifts center accurately in pixel space, preserving size and rotation.
    //    At z = 0.5 m, fx = 500 -> 1000 px/m.
    //    vel = (+0.5, +0.25, 0) m/s, dt = 0.04 s, gain = 0.7 -> lead = 0.028 s
    //    dx = +0.014 m (+14 px), dy = +0.007 m (-7 px in image y).
    {
        RoiPredictParams p;
        p.gain = 0.7;
        const TestRoi r = predict_roi(cam, base, palm, {0.5, 0.25, 0.0}, 0.04, p);
        CHECK(near(r.center[0], 514.0, 1e-3));
        CHECK(near(r.center[1], 493.0, 1e-3));
        CHECK(near(r.size, base.size) && near(r.rotation, base.rotation));
    }

    // 5. Elapsed time dt is capped at max_dt.
    {
        RoiPredictParams p;
        p.gain = 0.5;
        p.max_dt = 0.10;
        p.max_shift = 10.0;   // don't hit pixel cap here
        // vel = 0.2 m/s, capped lead = 0.10 * 0.5 = 0.05 s -> dx = 0.01 m -> +10 px
        const TestRoi r = predict_roi(cam, base, palm, {0.2, 0.0, 0.0}, 1.0, p);
        CHECK(near(r.center[0], 510.0, 1e-3));
    }

    // 6. Extreme velocity is capped at max_shift * roi.size.
    {
        RoiPredictParams p;
        p.gain = 1.0;
        p.max_shift = 0.35;   // 0.35 * 200 = 70 px
        const TestRoi r = predict_roi(cam, base, palm, {5.0, 0.0, 0.0}, 0.05, p);
        CHECK(near(r.center[0], 570.0, 1e-3));
        CHECK(near(r.center[1], 500.0, 1e-3));
    }

    // 7. Hand moving behind the camera returns unchanged ROI safely.
    {
        RoiPredictParams p;
        p.gain = 1.0;
        const V3 close_palm{0.0, 0.0, -0.1};
        const TestRoi r = predict_roi(cam, base, close_palm, {0.0, 0.0, 2.0}, 0.1, p);
        CHECK(near(r.center[0], base.center[0]) && near(r.center[1], base.center[1]));
    }

    // 8. Center is clamped to image bounds [0, width] x [0, height].
    {
        RoiPredictParams p;
        p.gain = 1.0;
        p.max_shift = 0.35;
        const TestRoi edge{{980.0, 10.0}, 200.0, 0.0};
        const TestRoi r = predict_roi(cam, edge, palm, {2.0, 2.0, 0.0}, 0.05, p);
        CHECK(r.center[0] <= cam.width && near(r.center[0], double(cam.width)));
        CHECK(r.center[1] >= 0.0 && near(r.center[1], 0.0));
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
