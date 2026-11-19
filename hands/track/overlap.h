// Two-hand overlap resilience for tracked views: handedness-aware duplicate / join /
// re-acquire checks, plus optional crop nudging and tightening away from the other hand.
#pragma once

#include "geom.h"

#include <algorithm>
#include <cmath>

struct OverlapParams {
    bool guard = false;         // handedness-aware duplicate, join, and re-acquire checks
    double nudge = 0.0;         // max crop center shift away from other hand (share of roi.size; e.g. 0.15)
    double tighten = 0.0;       // max crop size reduction when other hand enters crop (share; e.g. 0.15 -> 0.85x)
    double trigger_frac = 0.85; // act when other hand's projected palm is within trigger_frac * roi.size
    double side_diff = 0.5;     // |right_a - right_b| >= this means opposite hands
    double opp_dup_frac = 0.15; // opposite-side views only collide if mean landmark distance < this * hand_size
    double opp_clear_m = 0.03;  // opposite-side re-acquire clearance (m) when guard is on
};

inline bool sides_conflict(double right_a, double right_b, const OverlapParams &p) {
    return p.guard && std::fabs(right_a - right_b) >= p.side_diff;
}

// Whether two views in the same camera are duplicates of the same physical hand.
// Without guard (or when both views agree on handedness), matches the original same_hand rule.
// When guard is on and the two views strongly disagree on handedness, keep both unless their
// 21 landmarks sit almost directly on top of each other (same hand misclassified).
inline bool views_are_duplicate(const V2 a_pts[21], double a_right, const V2 b_pts[21], double b_right, double size,
                                const OverlapParams &p) {
    if (size <= 0) return false;
    double mean_d = 0;
    for (int i = 0; i < 21; ++i) mean_d += norm(a_pts[i] - b_pts[i]) / 21.0;
    if (sides_conflict(a_right, b_right, p)) return mean_d < p.opp_dup_frac * size;
    return norm(a_pts[9] - b_pts[9]) < 0.5 * size || mean_d < 0.25 * size;
}

// Nudge a tracked hand's crop center slightly away from the other hand's projected palm in the
// same camera, and optionally tighten the crop size modestly, to exclude the second hand before
// running the single-hand landmark model. No-op when the other hand is outside trigger_frac.
template <typename RoiType, typename CamType>
inline RoiType nudge_roi_away(const CamType &cam, RoiType roi, V2 other_uv, const OverlapParams &p) {
    if ((p.nudge <= 0 && p.tighten <= 0) || roi.size <= 0 || p.trigger_frac <= 0) return roi;
    const V2 diff = roi.center - other_uv;
    const double dist = norm(diff);
    const double trig = p.trigger_frac * roi.size;
    if (!std::isfinite(dist) || dist <= 1e-3 || dist >= trig) return roi;
    const double w = std::clamp((trig - dist) / trig, 0.0, 1.0);
    if (p.nudge > 0) {
        const double shift = std::clamp(p.nudge, 0.0, 0.35) * roi.size * w;
        roi.center[0] = std::clamp(roi.center[0] + (diff[0] / dist) * shift, 0.0, double(cam.width));
        roi.center[1] = std::clamp(roi.center[1] + (diff[1] / dist) * shift, 0.0, double(cam.height));
    }
    if (p.tighten > 0) {
        const double shrink = std::clamp(p.tighten, 0.0, 0.35) * w;
        roi.size *= (1.0 - shrink);
    }
    return roi;
}
