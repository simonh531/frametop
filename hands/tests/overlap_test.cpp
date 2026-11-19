// Unit tests for two-hand overlap resilience (track/overlap.h):
// no ncnn, no models, no recordings. Run via `make check` or `just test-c`.
#include "../track/overlap.h"

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

struct TestCam {
    int width = 1000, height = 1000;
};

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) <= eps; }

// Build a synthetic 21-point hand shifted by (dx, dy) around a center, with fingers spread.
void make_hand(V2 out[21], double cx, double cy, double spread = 40.0) {
    for (int i = 0; i < 21; ++i) {
        const double ang = (i / 21.0) * 6.283185307179586;
        const double r = (i == 9) ? 0.0 : spread * (0.4 + 0.6 * ((i % 4) / 3.0));
        out[i] = {cx + r * std::cos(ang), cy + r * std::sin(ang)};
    }
}

}  // namespace

int main() {
    const TestCam cam;

    // 1. Default (guard off): matches legacy same_hand behavior regardless of handedness.
    {
        OverlapParams p;
        V2 a[21], b[21];
        make_hand(a, 500, 500, 40);
        make_hand(b, 535, 500, 40);   // palm dist 35 < 0.5 * 100
        CHECK(views_are_duplicate(a, 0.05, b, 0.95, 100.0, p));
    }

    // 2. Guard on: opposite-side hands with palms 35% of hand_size apart are NOT duplicates.
    {
        OverlapParams p;
        p.guard = true;
        V2 a[21], b[21];
        make_hand(a, 500, 500, 40);
        make_hand(b, 535, 500, 40);   // mean_d = 35 >= 0.15 * 100
        CHECK(!views_are_duplicate(a, 0.05, b, 0.95, 100.0, p));
    }

    // 3. Guard on: same-side views at 35% of hand_size ARE still duplicates.
    {
        OverlapParams p;
        p.guard = true;
        V2 a[21], b[21];
        make_hand(a, 500, 500, 40);
        make_hand(b, 535, 500, 40);
        CHECK(views_are_duplicate(a, 0.90, b, 0.85, 100.0, p));
    }

    // 4. Guard on: opposite-side views whose 21 landmarks sit right on top of each other (< 0.15 * size)
    //    ARE still caught as duplicates (same physical hand misclassified as opposite sides).
    {
        OverlapParams p;
        p.guard = true;
        V2 a[21], b[21];
        make_hand(a, 500, 500, 40);
        make_hand(b, 508, 502, 40);   // mean_d ~ 8.2 < 15
        CHECK(views_are_duplicate(a, 0.05, b, 0.95, 100.0, p));
    }

    // 5. sides_conflict respects guard flag and threshold.
    {
        OverlapParams off;
        CHECK(!sides_conflict(0.05, 0.95, off));
        OverlapParams on;
        on.guard = true;
        CHECK(sides_conflict(0.1, 0.9, on));
        CHECK(!sides_conflict(0.4, 0.7, on));   // ambiguous (diff 0.3 < 0.5)
    }

    // 6. nudge_roi_away: no-op when off or when other hand is outside trigger_frac.
    {
        OverlapParams p;
        p.nudge = 0.15;
        p.tighten = 0.15;
        p.trigger_frac = 0.85;
        const TestRoi base{{500.0, 500.0}, 200.0, 0.25};
        // Other hand at distance 180 >= 0.85 * 200 (170): untouched.
        const TestRoi far = nudge_roi_away(cam, base, {680.0, 500.0}, p);
        CHECK(near(far.center[0], 500.0) && near(far.center[1], 500.0));
        CHECK(near(far.size, 200.0) && near(far.rotation, 0.25));
    }

    // 7. nudge_roi_away: pushes center away from nearby other hand and tightens size smoothly.
    {
        OverlapParams p;
        p.nudge = 0.20;
        p.tighten = 0.15;
        p.trigger_frac = 0.85;   // trig = 170 px for size 200
        const TestRoi base{{500.0, 500.0}, 200.0, 0.25};
        // Other hand 85 px to the right -> w = (170 - 85) / 170 = 0.5
        // shift = 0.20 * 200 * 0.5 = 20 px left -> center.x = 480
        // size = 200 * (1 - 0.15 * 0.5) = 185
        const TestRoi r = nudge_roi_away(cam, base, {585.0, 500.0}, p);
        CHECK(near(r.center[0], 480.0, 1e-3));
        CHECK(near(r.center[1], 500.0, 1e-3));
        CHECK(near(r.size, 185.0, 1e-3));
        CHECK(near(r.rotation, 0.25));
    }

    // 8. nudge_roi_away: coincident centers (no well-defined direction) leave ROI untouched safely.
    {
        OverlapParams p;
        p.nudge = 0.20;
        p.tighten = 0.15;
        const TestRoi base{{500.0, 500.0}, 200.0, 0.0};
        const TestRoi r = nudge_roi_away(cam, base, {500.0, 500.0}, p);
        CHECK(near(r.center[0], 500.0) && near(r.center[1], 500.0) && near(r.size, 200.0));
    }

    // 9. nudge_roi_away: clamps nudged center to camera bounds.
    {
        OverlapParams p;
        p.nudge = 0.30;
        const TestRoi edge{{10.0, 500.0}, 200.0, 0.0};
        const TestRoi r = nudge_roi_away(cam, edge, {50.0, 500.0}, p);
        CHECK(near(r.center[0], 0.0));
        CHECK(near(r.center[1], 500.0));
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
