#!/usr/bin/env python3
"""Measure hand-tracking 'freak-out' failures from ft-handreplay --diag JSONL dumps.

usage:
  python3 hands/tools/freakout_report.py DIAG [DIAG...] [--takes-dir DIR] [--top N] [--json OUT]

DIAG comes from `hands/build/ft-handreplay TAKE --diag DIAG`. When --takes-dir (or a matching
prompts.jsonl beside the take) is provided, processed sets are bucketed by prompt segment
(normal: pinch-tap/pinch-drag/grab/screen-point; cross; overlap; near-face) via dq_ns.

Per hand update, four orthogonal freak-out flags are evaluated:
  geom_explode   fit_rms > 60 mm or max raw bone-length error > 2.5x calibrated length
  orient_flip    palm normal rotates > 75 deg in a single step (< 0.25 s)
  teleport_snap  published palm jumps > 100 mm or finger-relative tip jumps > 100 mm in one step
  side_conflict  same track ID flips side, skeleton fit flips side, or views disagree on right_score > 0.35

Beside freak-out rates, the report also prints two-hand presence (%) and dup/split churn per
100 sets so improvements can't be gamed by simply hiding or dropping overlapping hands.
"""
import argparse
import collections
import glob
import json
import math
import os


NORMAL_POSES = {"pinch-tap", "pinch-drag", "grab", "screen-point"}


def _vsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vnorm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _ang_deg(a, b):
    na, nb = _vnorm(a), _vnorm(b)
    if na < 1e-6 or nb < 1e-6:
        return None
    c = max(-1.0, min(1.0, _vdot(a, b) / (na * nb)))
    return math.degrees(math.acos(c))


def _pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def load_prompts(take_dir):
    path = os.path.join(take_dir, "prompts.jsonl")
    if not os.path.exists(path):
        return []
    events = []
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if o.get("event") in ("prompt", "ready", "pause", "done"):
                events.append((o["t"], o["event"], o.get("pose", "")))
    windows = []
    for i, (t, ev, pose) in enumerate(events):
        if ev == "prompt":
            t_end = events[i + 1][0] if i + 1 < len(events) else t + 10_000_000_000
            windows.append((t, t_end, pose))
    return windows


def label_pose(dq_ns, windows):
    if not windows:
        return "all"
    for a, b, pose in windows:
        if a <= dq_ns < b:
            return pose
    return "between"


def classify_hand(rec):
    flags = []
    coasting = rec.get("coasting", False)
    # When coasting, the corrupted raw measurement is rejected and the published skeleton stays valid.
    if not coasting and (
        (rec["fit_rms_mm"] is not None and rec["fit_rms_mm"] > 60.0)
        or (rec["bone_err_max"] is not None and rec["bone_err_max"] > 2.5)
    ):
        flags.append("geom_explode")
    if rec["norm_deg"] is not None and rec["norm_deg"] > 75.0:
        flags.append("orient_flip")
    if (rec["fit_jump_mm"] is not None and rec["fit_jump_mm"] > 100.0) or (
        rec["tip_rel_jump_mm"] is not None and rec["tip_rel_jump_mm"] > 100.0
    ):
        flags.append("teleport_snap")
    if not coasting and (rec["side_flipped_same_id"] or rec["fit_flipped"] or rec["view_side_spread"] > 0.35):
        flags.append("side_conflict")
    return flags


def severity_score(rec):
    s = 0.0
    if rec["fit_rms_mm"] is not None:
        s += max(0.0, rec["fit_rms_mm"] - 30.0) / 15.0
    if rec["bone_err_max"] is not None:
        s += max(0.0, rec["bone_err_max"] - 1.5) * 1.5
    if rec["fit_jump_mm"] is not None:
        s += max(0.0, rec["fit_jump_mm"] - 50.0) / 25.0
    if rec["norm_deg"] is not None:
        s += max(0.0, rec["norm_deg"] - 40.0) / 25.0
    if rec["tip_rel_jump_mm"] is not None:
        s += max(0.0, rec["tip_rel_jump_mm"] - 50.0) / 25.0
    if rec["view_side_spread"] > 0.35:
        s += 2.0 * rec["view_side_spread"]
    if rec["side_flipped_same_id"]:
        s += 2.0
    if rec["fit_flipped"]:
        s += 1.5
    return s


def analyze_file(diag_path, take_dir=None, name=None):
    cid = name or os.path.splitext(os.path.basename(diag_path))[0]
    windows = load_prompts(take_dir) if take_dir else []
    prev_by_side = {}
    prev_by_id = {}
    hand_recs = []
    set_recs = []

    with open(diag_path) as f:
        for line in f:
            row = json.loads(line)
            pose = label_pose(row["dq_ns"], windows)
            evs = row["events"]
            hands = row["hands"]
            ts = row["ts"]

            ev_tokens = [e.split()[0] for e in evs if e]
            same_side_dup = len(hands) >= 2 and len({h["side"] for h in hands}) < len(hands)
            has_both = {"L", "R"}.issubset({h["side"] for h in hands})

            # Pairwise 3D distance for context
            other_3d = {}
            for i, h in enumerate(hands):
                dmin = float("inf")
                for j, o in enumerate(hands):
                    if i != j:
                        dmin = min(dmin, _vnorm(_vsub(h["palm_raw"], o["palm_raw"])))
                other_3d[h["id"]] = dmin if math.isfinite(dmin) else None

            next_prev_by_side = {}
            next_prev_by_id = {}
            set_any_flag = False
            set_severe = False

            for h in hands:
                side = h["side"]
                hid = h["id"]

                bone_err_max = None
                if h["bone_ref"] and h["bone_len"]:
                    errs = [abs(bl / br - 1.0) for bl, br in zip(h["bone_len"], h["bone_ref"]) if bl and br and br > 1e-4]
                    if errs:
                        bone_err_max = max(errs)

                v_rights = [v["right"] for v in h["views"]]
                view_side_spread = (max(v_rights) - min(v_rights)) if len(v_rights) >= 2 else 0.0

                pid = prev_by_id.get(hid)
                pside = prev_by_side.get(side)
                p = pid if (pid is not None and 0.005 < (ts - pid["ts"]) < 0.25) else pside

                fit_jump_mm = norm_deg = tip_rel_jump_mm = None
                id_changed = False
                side_flipped_same_id = False

                if p is not None and 0.005 < (ts - p["ts"]) < 0.25:
                    fit_jump_mm = _vnorm(_vsub(h["palm_fit"], p["palm_fit"])) * 1000.0
                    norm_deg = _ang_deg(h["palm_normal"], p["palm_normal"])
                    id_changed = pside is not None and 0.005 < (ts - pside["ts"]) < 0.25 and hid != pside["id"]
                    tip_jumps = [
                        _vnorm(_vsub(_vsub(tn, h["palm_fit"]), _vsub(tp, p["palm_fit"]))) * 1000.0
                        for tn, tp in zip(h["tips_fit"], p["tips_fit"])
                    ]
                    tip_rel_jump_mm = max(tip_jumps) if tip_jumps else None

                if pid is not None and 0.005 < (ts - pid["ts"]) < 0.25:
                    side_flipped_same_id = side != pid["side"]

                rec = {
                    "cid": cid,
                    "set": row["set"],
                    "ts": ts,
                    "pose": pose,
                    "id": hid,
                    "side": side,
                    "nviews": h["nviews"],
                    "residual_mm": h["residual"] * 1000.0 if h["residual"] is not None else None,
                    "fit_rms_mm": h["fit_rms"] * 1000.0 if h["fit_rms"] is not None else None,
                    "bone_err_max": bone_err_max,
                    "fit_flipped": h["fit_flipped"],
                    "coasting": h.get("coasting", False),
                    "view_side_spread": view_side_spread,
                    "fit_jump_mm": fit_jump_mm,
                    "norm_deg": norm_deg,
                    "tip_rel_jump_mm": tip_rel_jump_mm,
                    "id_changed": id_changed,
                    "side_flipped_same_id": side_flipped_same_id,
                    "same_side_dup": same_side_dup,
                    "other_3d_m": other_3d[hid],
                    "events": evs,
                }
                rec["flags"] = classify_hand(rec)
                rec["score"] = severity_score(rec)
                if rec["flags"]:
                    set_any_flag = True
                if len(rec["flags"]) >= 2:
                    set_severe = True
                hand_recs.append(rec)

                snap = {
                    "ts": ts, "id": hid, "side": side,
                    "palm_fit": h["palm_fit"], "palm_normal": h["palm_normal"], "tips_fit": h["tips_fit"],
                }
                next_prev_by_id[hid] = snap
                if side not in next_prev_by_side:
                    next_prev_by_side[side] = snap

            prev_by_id.update(next_prev_by_id)
            prev_by_side.update(next_prev_by_side)

            set_recs.append({
                "cid": cid,
                "set": row["set"],
                "ts": ts,
                "pose": pose,
                "nhands": len(hands),
                "has_both": has_both,
                "same_side_dup": same_side_dup,
                "any_flag": set_any_flag,
                "severe": set_severe,
                "dups": ev_tokens.count("dup"),
                "splits": ev_tokens.count("split"),
                "lost": ev_tokens.count("lost"),
            })

    return hand_recs, set_recs


def summarize(hand_recs, set_recs, poses=None):
    hs = [r for r in hand_recs if poses is None or r["pose"] in poses]
    ss = [r for r in set_recs if poses is None or r["pose"] in poses]
    nh, ns = len(hs), len(ss)
    if not ns:
        return None
    fc = collections.Counter()
    any_h = sev_h = 0
    for r in hs:
        if r["flags"]:
            any_h += 1
        if len(r["flags"]) >= 2:
            sev_h += 1
        for fl in r["flags"]:
            fc[fl] += 1
    return {
        "sets": ns,
        "hands": nh,
        "both_pct": 100.0 * sum(1 for s in ss if s["has_both"]) / ns,
        "set_freak_pct": 100.0 * sum(1 for s in ss if s["any_flag"]) / ns,
        "set_severe_pct": 100.0 * sum(1 for s in ss if s["severe"]) / ns,
        "dup_per_100": 100.0 * sum(s["dups"] for s in ss) / ns,
        "split_per_100": 100.0 * sum(s["splits"] for s in ss) / ns,
        "hand_any_pct": 100.0 * any_h / max(nh, 1),
        "hand_2p_pct": 100.0 * sev_h / max(nh, 1),
        "geom_pct": 100.0 * fc["geom_explode"] / max(nh, 1),
        "orient_pct": 100.0 * fc["orient_flip"] / max(nh, 1),
        "telep_pct": 100.0 * fc["teleport_snap"] / max(nh, 1),
        "side_pct": 100.0 * fc["side_conflict"] / max(nh, 1),
        "coast_pct": 100.0 * sum(1 for r in hs if r.get("coasting")) / max(nh, 1),
        "ndeg_p95": _pct([r["norm_deg"] for r in hs if r["norm_deg"] is not None], 0.95),
        "fjump_p95": _pct([r["fit_jump_mm"] for r in hs if r["fit_jump_mm"] is not None], 0.95),
        "rms_p95": _pct([r["fit_rms_mm"] for r in hs if r["fit_rms_mm"] is not None], 0.95),
    }


def find_takes(takes_dir):
    out = {}
    if not takes_dir:
        return out
    for t in sorted(glob.glob(os.path.join(takes_dir, "**", "takes", "03-gestures"), recursive=True)):
        cid = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(t))))[:8]
        out[cid] = t
    return out


def print_row(label, s):
    print(
        f"{label:15s} {s['sets']:5d} {s['both_pct']:5.1f}% | "
        f"{s['set_freak_pct']:5.1f}% {s['set_severe_pct']:5.1f}% | "
        f"{s['geom_pct']:5.1f}% {s['orient_pct']:5.1f}% {s['telep_pct']:5.1f}% {s['side_pct']:5.1f}% {s['coast_pct']:5.1f}% | "
        f"{s['dup_per_100']:5.1f} {s['split_per_100']:5.1f} | "
        f"{s['ndeg_p95']:5.1f} {s['fjump_p95']:5.1f} {s['rms_p95']:5.1f}"
    )


def main():
    ap = argparse.ArgumentParser(description="Score hand-tracking freak-outs from ft-handreplay --diag JSONL dumps.")
    ap.add_argument("diags", nargs="+", help="One or more --diag JSONL files")
    ap.add_argument("--takes-dir", default="hands/dumps/hf/contributions", help="Root dir to auto-match 03-gestures prompts.jsonl")
    ap.add_argument("--take", default=None, help="Explicit take dir (when passing a single DIAG)")
    ap.add_argument("--top", type=int, default=0, help="Print top N worst cross/overlap frames")
    ap.add_argument("--json", dest="json_out", default=None, help="Write summary JSON to this path")
    args = ap.parse_args()

    takes = find_takes(args.takes_dir) if not args.take else {}
    all_hands, all_sets = [], []
    per_take = {}

    for dp in sorted(args.diags):
        cid = os.path.splitext(os.path.basename(dp))[0]
        td = args.take or takes.get(cid)
        hs, ss = analyze_file(dp, td, cid)
        all_hands.extend(hs)
        all_sets.extend(ss)
        per_take[cid] = (hs, ss)

    hdr = (
        f"{'segment':15s} {'sets':>5s} {'both':>6s} | "
        f"{'setAny':>6s} {'set2+':>6s} | "
        f"{'geom':>6s} {'orient':>6s} {'telep':>6s} {'side':>6s} {'coast':>6s} | "
        f"{'dup/c':>5s} {'spl/c':>5s} | "
        f"{'n95°':>5s} {'j95mm':>5s} {'r95mm':>5s}"
    )
    print("=== Summary by prompt segment ===")
    print(hdr)
    has_prompts = any(s["pose"] != "all" for s in all_sets)
    summary_out = {"segments": {}, "per_take_cross_overlap": {}}
    if has_prompts:
        groups = [
            ("normal", NORMAL_POSES),
            ("cross", {"cross"}),
            ("overlap", {"overlap"}),
            ("cross+overlap", {"cross", "overlap"}),
            ("near-face", {"near-face"}),
        ]
    else:
        groups = [("all", None)]

    for gname, poses in groups:
        s = summarize(all_hands, all_sets, poses)
        if s:
            print_row(gname, s)
            summary_out["segments"][gname] = s

    if len(per_take) > 1 and has_prompts:
        print("\n=== Per-take breakdown (cross + overlap) ===")
        print(hdr)
        for cid, (hs, ss) in sorted(per_take.items()):
            s = summarize(hs, ss, {"cross", "overlap"})
            if s:
                print_row(cid, s)
                summary_out["per_take_cross_overlap"][cid] = s

    if args.top > 0:
        pool = [r for r in all_hands if r["pose"] in ("cross", "overlap")] if has_prompts else list(all_hands)
        pool.sort(key=lambda r: r["score"], reverse=True)
        print(f"\n=== Top {args.top} worst frames ===")
        for r in pool[: args.top]:
            fl = ",".join(r["flags"]) or "-"
            print(
                f"  {r['cid']} set={r['set']:4d} ts={r['ts']:6.2f} {r['pose']:7s} id={r['id']:2d}({r['side']}) nv={r['nviews']} "
                f"score={r['score']:5.1f} [{fl}] rms={r['fit_rms_mm'] or -1:5.1f} bmax={r['bone_err_max'] or -1:4.1f} "
                f"fjump={r['fit_jump_mm'] or -1:5.1f} ndeg={r['norm_deg'] or -1:5.1f} "
                f"tjump={r['tip_rel_jump_mm'] or -1:5.1f} vsp={r['view_side_spread']:.2f}"
            )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary_out, f, indent=2)


if __name__ == "__main__":
    main()
