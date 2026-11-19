// Velocity-based active ROI prediction for tracked hand views.
// Shifts a view's next landmark crop forward in image space using the hand's 3D
// palm velocity projected through the camera, so fast-moving hands don't outrun
// their previous-frame crop and trigger expensive palm-detector searches.
#pragma once

#include "geom.h"

#include <algorithm>
#include <cmath>

struct RoiPredictParams {
    double gain = 0.0;        // fraction of predicted displacement to apply (0: off; e.g. 0.7 live)
    double max_dt = 0.15;     // cap elapsed time (s) so stalls/gaps don't throw the crop away
    double max_shift = 0.35;  // cap pixel shift as a share of roi.size (never jump off the hand)
    double min_speed = 0.05;  // ignore slow/noisy drift under this speed (m/s)
};

// Advance a crop's center in image space along the hand's 3D velocity.
// RoiType: any struct with V2 center and double size (e.g. Roi in nets.h).
// CamType: any camera with width, height, and V2 project(V3 head, double *depth) const.
template <typename RoiType, typename CamType>
inline RoiType predict_roi(const CamType &cam, RoiType roi, V3 palm, V3 vel, double dt, const RoiPredictParams &p) {
    if (p.gain <= 0 || dt <= 0 || roi.size <= 0) return roi;
    const double speed = norm(vel);
    if (!std::isfinite(speed) || speed < p.min_speed) return roi;
    const double lead = std::min(dt, p.max_dt) * p.gain;
    const V3 ahead = palm + vel * lead;
    double z0 = 0, z1 = 0;
    const V2 uv0 = cam.project(palm, &z0);
    const V2 uv1 = cam.project(ahead, &z1);
    if (z0 <= 0.05 || z1 <= 0.05) return roi;
    V2 duv = uv1 - uv0;
    const double d = norm(duv);
    if (!std::isfinite(d) || d <= 0) return roi;
    const double cap = p.max_shift * roi.size;
    if (d > cap) duv = duv * (cap / d);
    roi.center[0] = std::clamp(roi.center[0] + duv[0], 0.0, double(cam.width));
    roi.center[1] = std::clamp(roi.center[1] + duv[1], 0.0, double(cam.height));
    return roi;
}
