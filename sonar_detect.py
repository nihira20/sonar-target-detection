"""
Speckle-Adaptive, Training-Free Sonar Target Detection
with Range-Consistent Shadow Validation

Pipeline
  1. Estimate speckle level (coefficient of variation on homogeneous blocks)
  2. Pick and parameterize a despeckling filter from that estimate
  3. Flatten range gain, robust z-score map (median / MAD)
  4. Highlight mask + shadow mask, morphological cleanup
  5. Candidates from highlight blobs, validated by a shadow found on the
     range axis (left/right or up/down of the object)
  6. Output best box, optional evaluation (IoU, P, R, F1) and ablation

Usage (script):
  python sonar_detect.py --image sonar.png --out result.png
  python sonar_detect.py --img_dir imgs/ --gt gt.csv --ablation
    gt.csv columns: filename,x1,y1,x2,y2   (one row per box)

Usage (Colab / notebook):
  from sonar_detect import *
  res = detect(load_gray("sonar.png"), Cfg())
  show(res, save="result.png")
"""
import os
import csv
import argparse
from dataclasses import dataclass, replace

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class Cfg:
    filter_mode: str = "adaptive"   # adaptive | none | bilateral | lee5 | lee9 | nlm
    use_shadow: bool = True         # shadow validation on/off (ablation)
    range_axis: str = "horizontal"  # direction shadows are cast: horizontal | vertical
    kh: float = 2.0                 # highlight z threshold
    ks: float = 1.0                 # shadow z threshold (positive number)
    dens_k: int = 21                # density window (pixels at ~340 px image height; auto-scaled)
    dens_thr: float = 0.35          # seed density: core of an object
    dens_lo: float = 0.18           # grow region down to this density (hysteresis)
    min_area_frac: float = 0.003    # min object area / image area
    max_dim_frac: float = 0.9       # reject boxes spanning >90% of width or height
    refine: bool = False            # grow the box over faint parts of the object (thin hull edges, rails)
    kh_lo: float = 1.2              # looser highlight threshold used only for box refinement
    refine_gap: float = 0.25        # max gap to a faint fragment, as a fraction of current box size
    group_gap: float = 0.35         # merge fragments closer than this * size of the larger fragment (0 = off)
    shadow_len_factor: float = 1.0  # shadow search length = factor * object width
    min_shadow: float = 0.20        # min shadow fraction to call a candidate validated
    shadow_weight: float = 2.0      # score boost from shadow evidence
    topk: int = 1                   # detections to return
    block: int = 16                 # block size for speckle estimation


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def add_speckle(img, looks=4, seed=0):
    """Multiplicative gamma speckle (stress test). Fewer looks = stronger speckle."""
    rng = np.random.default_rng(seed)
    n = rng.gamma(shape=looks, scale=1.0 / looks, size=img.shape)
    return np.clip(img.astype(np.float32) * n, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Step 1: speckle estimation
# ----------------------------------------------------------------------------
def estimate_speckle_cv(img, block=16, keep=0.15):
    """
    Coefficient of variation (std/mean) over the most homogeneous bright-enough
    blocks. Fully developed speckle gives a roughly constant CV; stronger
    speckle (fewer looks) gives a larger CV.
    """
    f = img.astype(np.float32)
    H, W = f.shape
    H2, W2 = (H // block) * block, (W // block) * block
    b = f[:H2, :W2].reshape(H2 // block, block, W2 // block, block)
    b = b.swapaxes(1, 2).reshape(-1, block * block)
    m, s = b.mean(1), b.std(1)
    cv = s / (m + 1e-6)
    valid = m > np.percentile(m, 30)          # ignore black shadow / nadir blocks
    cv = cv[valid]
    if cv.size == 0:
        return 0.5
    cv = np.sort(cv)[: max(5, int(keep * cv.size))]
    return float(np.clip(np.median(cv), 0.1, 1.2))


# ----------------------------------------------------------------------------
# Step 2: filter bank
# ----------------------------------------------------------------------------
def lee_filter(img, size, cu):
    """Lee filter for multiplicative noise; cu = noise coefficient of variation."""
    f = img.astype(np.float32)
    m = cv2.blur(f, (size, size))
    v = cv2.blur(f * f, (size, size)) - m * m
    ci2 = np.maximum(v, 0) / (m * m + 1e-6)
    w = np.clip(1.0 - (cu ** 2) / (ci2 + 1e-6), 0.0, 1.0)
    return np.clip(m + w * (f - m), 0, 255).astype(np.uint8)


def nlm_filter(img, cu):
    h = float(np.clip(25 * cu, 8, 25))
    return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=7, searchWindowSize=21)


def choose_filter(cu):
    """Heuristic mapping from speckle level to filter. Tune these cut-offs on a
    validation split (this mapping is itself an experiment for the paper)."""
    if cu < 0.28:
        return "bilateral"
    if cu < 0.45:
        return "lee5"
    if cu < 0.65:
        return "lee9"
    return "nlm"


def apply_filter(img, name, cu):
    if name == "none":
        return img.copy()
    if name == "bilateral":
        return cv2.bilateralFilter(img, 7, 40, 10)
    if name == "lee5":
        return lee_filter(img, 5, cu)
    if name == "lee9":
        return lee_filter(img, 9, cu)
    if name == "nlm":
        return nlm_filter(img, cu)
    raise ValueError(name)

# ==================================================================================== END OF PART N ====================================================================================

# ----------------------------------------------------------------------------
# Step 3 and 4: normalization and masks
# ----------------------------------------------------------------------------
def zscore_map(f_img):
    f = f_img.astype(np.float32)
    H, W = f.shape
    clipped = np.minimum(f, np.percentile(f, 90))        # keep object from biasing background
    bg = cv2.GaussianBlur(clipped, (0, 0), sigmaX=min(H, W) / 6.0)
    ratio = f / (bg + 1.0)                                # removes range gain / slow trends
    med = np.median(ratio)
    mad = np.median(np.abs(ratio - med)) * 1.4826
    return (ratio - med) / (mad + 1e-6)


def ellipse(k):
    k = int(k) | 1                       # force odd
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def rel(n, shape, ref=340.0):
    """Scale a pixel size tuned for a ~340 px tall image to this image's resolution."""
    return max(3, int(round(n * min(shape) / ref))) | 1


def remove_small(mask, min_px):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out[lab == i] = 255
    return out


def make_masks(z, cfg):
    H, W = z.shape
    min_px = int(cfg.min_area_frac * H * W)
    hi_raw = ((z > cfg.kh) * 255).astype(np.uint8)
    hi_raw = cv2.morphologyEx(hi_raw, cv2.MORPH_OPEN, ellipse(rel(3, z.shape)))
    sh = ((z < -cfg.ks) * 255).astype(np.uint8)
    sh = cv2.morphologyEx(sh, cv2.MORPH_OPEN, ellipse(rel(5, z.shape)))
    sh = cv2.morphologyEx(sh, cv2.MORPH_CLOSE, ellipse(rel(9, z.shape)))
    sh = remove_small(sh, max(min_px // 2, 50))
    # density map: object highlights are dense, seabed speckle clutter is sparse
    dk = rel(cfg.dens_k, z.shape)
    dens = cv2.blur((hi_raw > 0).astype(np.float32), (dk, dk))
    seeds = ((dens > cfg.dens_thr) * 255).astype(np.uint8)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, ellipse(rel(7, z.shape)))
    seeds = remove_small(seeds, min_px)
    low = ((dens > cfg.dens_lo) * 255).astype(np.uint8)
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, ellipse(rel(7, z.shape)))
    # hysteresis: keep low-density regions that contain a dense seed
    n, lab = cv2.connectedComponents(low, connectivity=8)
    keep = np.unique(lab[seeds > 0])
    hi_merged = np.isin(lab, keep[keep > 0]).astype(np.uint8) * 255
    return hi_raw, hi_merged, sh

# ==================================================================================== END OF PART S ====================================================================================
# ----------------------------------------------------------------------------
# Step 5: candidates and shadow validation
# ----------------------------------------------------------------------------
def shadow_evidence(sh, comp, box, cfg):
    """
    Shadow pixels that touch the object along the range axis (both sides),
    measured against the object size. Works when the shadow lies outside the
    object box (ships, mines) and when it falls inside it (aircraft wings).
    """
    H, W = sh.shape
    x, y, w, h = box
    if cfg.range_axis == "horizontal":
        L = max(int(cfg.shadow_len_factor * w), 15)
        kernels = {"right": (L, 3, "r"), "left": (L, 3, "l")}
    else:
        L = max(int(cfg.shadow_len_factor * h), 15)
        kernels = {"down": (3, L, "d"), "up": (3, L, "u")}
    comp_u8 = comp.astype(np.uint8) * 255
    best = (0.0, None, None)
    for side, (kw, kh_, d) in kernels.items():
        k = np.ones((kh_, 2 * kw + 1) if cfg.range_axis == "horizontal" else (2 * kw + 1, kh_), np.uint8)
        # one-sided dilation: anchor at the edge so growth goes one way only
        if cfg.range_axis == "horizontal":
            k = np.ones((kh_, kw + 1), np.uint8)
            anchor = (0, kh_ // 2) if d == "r" else (kw, kh_ // 2)
        else:
            k = np.ones((kw + 1, kh_), np.uint8) if False else np.ones((kh_ + 0, kw + 0), np.uint8)
            k = np.ones((L + 1, 3), np.uint8)
            anchor = (1, 0) if d == "d" else (1, L)
        grown = cv2.dilate(comp_u8, k, anchor=anchor)
        adj = (sh > 0) & (grown > 0) & (comp_u8 == 0)
        area = float(adj.sum())
        ev = min(1.0, area / (0.3 * comp.sum() + 1e-6))
        if ev > best[0]:
            ys, xs = np.nonzero(adj)
            zone = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if xs.size else None
            best = (ev, side, zone)
    return best


def shadow_box(sh, zone):
    return zone


def _gap(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def group_fragments(frags, gap_frac):
    """Single-linkage grouping: one physical object often breaks into several
    highlight blobs (wing, fuselage, tail). Merge blobs that are close
    relative to the size of the larger blob."""
    groups = [dict(boxes=[f["box"]], mask=f["mask"].copy(), area=f["area"]) for f in frags]
    changed = gap_frac > 0
    while changed:
        changed = False
        groups.sort(key=lambda g: g["area"], reverse=True)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                bi = _union(groups[i]["boxes"])
                bj = _union(groups[j]["boxes"])
                size = max(bi[2] - bi[0], bi[3] - bi[1], bj[2] - bj[0], bj[3] - bj[1])
                if _gap(bi, bj) <= gap_frac * size:
                    groups[i]["boxes"] += groups[j]["boxes"]
                    groups[i]["mask"] |= groups[j]["mask"]
                    groups[i]["area"] += groups[j]["area"]
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return groups


def _union(boxes):
    b = np.array(boxes)
    return (int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max()))


def extract_candidates(z, hi_raw, hi_merged, sh, cfg):
    H, W = z.shape
    min_px = cfg.min_area_frac * H * W
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hi_merged, connectivity=8)
    frags = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_px:
            continue
        comp = lab == i
        raw_pix = comp & (hi_raw > 0)
        if raw_pix.sum() < 20:
            continue
        ys, xs = np.nonzero(raw_pix)                        # tight box from raw highlight pixels
        tight = (int(np.percentile(xs, 1)), int(np.percentile(ys, 1)),
                 int(np.percentile(xs, 99)), int(np.percentile(ys, 99)))
        frags.append(dict(box=tight, mask=comp, area=int(area)))

    cands = []
    for g in group_fragments(frags, cfg.group_gap):
        x1, y1, x2, y2 = _union(g["boxes"])
        w, h = x2 - x1, y2 - y1
        if w > cfg.max_dim_frac * W or h > cfg.max_dim_frac * H:
            continue
        comp = g["mask"]
        raw_pix = comp & (hi_raw > 0)
        hl_strength = float(z[raw_pix].mean())
        hl_score = hl_strength * np.sqrt(g["area"] / (H * W))
        frac, side, zone = shadow_evidence(sh, comp, (x1, y1, w, h), cfg)
        validated = frac >= cfg.min_shadow
        score = hl_score * (1.0 + cfg.shadow_weight * frac) if cfg.use_shadow else hl_score
        cands.append(dict(
            box=(x1, y1, x2, y2), score=float(score), hl_strength=hl_strength,
            shadow_frac=frac, shadow_side=side, validated=bool(validated),
            shadow_box=shadow_box(sh, zone), n_fragments=len(g["boxes"]),
        ))
    key = (lambda c: (c["validated"], c["score"])) if cfg.use_shadow else (lambda c: c["score"])
    cands.sort(key=key, reverse=True)
    return cands

# ==================================================================================== END OF PART R ====================================================================================
def refine_box(z, box, cfg):
    """
    Stage 2: the strict masks find the object core; faint parts (thin hull
    edges, rails, debris) only show up at a looser threshold. Grow the core box
    over nearby loose-threshold fragments, ignoring small blobs on the border.
    """
    H, W = z.shape
    lo = ((z > cfg.kh_lo) * 255).astype(np.uint8)
    lo = cv2.morphologyEx(lo, cv2.MORPH_OPEN, ellipse(rel(3, z.shape)))
    dk = rel(cfg.dens_k, z.shape)
    dens = cv2.blur((lo > 0).astype(np.float32), (dk, dk))
    m = ((dens > cfg.dens_thr) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse(rel(7, z.shape)))
    m = remove_small(m, int(0.5 * cfg.min_area_frac * H * W))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    frags = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        frags.append((int(x), int(y), int(x + w), int(y + h)))
    cur = tuple(box)
    changed = True
    while changed:
        changed = False
        cw, ch = cur[2] - cur[0], cur[3] - cur[1]
        for fb in list(frags):
            on_border = fb[0] <= 1 or fb[1] <= 1 or fb[2] >= W - 2 or fb[3] >= H - 2
            small = (fb[2] - fb[0]) * (fb[3] - fb[1]) < 0.1 * cw * ch
            if on_border and small:
                frags.remove(fb)
                continue
            if _gap(cur, fb) <= cfg.refine_gap * max(cw, ch):
                new = (min(cur[0], fb[0]), min(cur[1], fb[1]), max(cur[2], fb[2]), max(cur[3], fb[3]))
                frags.remove(fb)
                if new != cur:
                    cur, changed = new, True
                break
    return cur


# ----------------------------------------------------------------------------
# Full detector
# ----------------------------------------------------------------------------
def detect(img, cfg=Cfg()):
    cu = estimate_speckle_cv(img, cfg.block)
    name = choose_filter(cu) if cfg.filter_mode == "adaptive" else cfg.filter_mode
    filt = apply_filter(img, name, cu)
    z = zscore_map(filt)
    hi_raw, hi_merged, sh = make_masks(z, cfg)
    cands = extract_candidates(z, hi_raw, hi_merged, sh, cfg)
    for c in cands[: cfg.topk]:
        c["box_core"] = c["box"]
        if cfg.refine:
            c["box"] = refine_box(z, c["box"], cfg)
    return dict(img=img, filt=filt, z=z, hi=hi_merged, sh=sh, cands=cands,
                dets=cands[: cfg.topk], speckle_cv=cu, filter_name=name)


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def draw(res, gt=None):
    out = cv2.cvtColor(res["img"], cv2.COLOR_GRAY2BGR)
    for d in res["dets"]:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tag = "object" if d["validated"] else "object (no shadow)"
        cv2.putText(out, f"{tag} {d['score']:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if d["shadow_box"] is not None:
            sx1, sy1, sx2, sy2 = d["shadow_box"]
            cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (255, 255, 0), 1)
    if gt is not None:
        for (x1, y1, x2, y2) in gt:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def show(res, gt=None, save=None):
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax[0, 0].imshow(res["img"], cmap="gray"); ax[0, 0].set_title("Original")
    ax[0, 1].imshow(res["filt"], cmap="gray")
    ax[0, 1].set_title(f"Despeckled: {res['filter_name']} (speckle CV={res['speckle_cv']:.2f})")
    ax[0, 2].imshow(np.clip(res["z"], -4, 6), cmap="magma"); ax[0, 2].set_title("Robust z-map")
    ax[1, 0].imshow(res["hi"], cmap="gray"); ax[1, 0].set_title("Highlight mask (merged)")
    ax[1, 1].imshow(res["sh"], cmap="gray"); ax[1, 1].set_title("Shadow mask")
    ax[1, 2].imshow(draw(res, gt)); ax[1, 2].set_title("Detection (green=object, cyan=shadow)")
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=130)
    plt.show()


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(csv_path):
    gt = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            gt.setdefault(r["filename"], []).append(
                tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
    return gt


def evaluate(img_dir, gt_csv, cfg=Cfg(), looks=None, iou_thr=0.5):
    gt = load_gt(gt_csv)
    tp = fp = fn = 0
    ious, t_total = [], 0.0
    for fn_name, boxes in gt.items():
        img = load_gray(os.path.join(img_dir, fn_name))
        if looks:
            img = add_speckle(img, looks)
        t0 = cv2.getTickCount()
        res = detect(img, cfg)
        t_total += (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        used = set()
        for d in res["dets"]:
            best, bj = 0.0, -1
            for j, g in enumerate(boxes):
                if j in used:
                    continue
                v = iou(d["box"], g)
                if v > best:
                    best, bj = v, j
            if best >= iou_thr:
                tp += 1; used.add(bj); ious.append(best)
            else:
                fp += 1
        fn += len(boxes) - len(used)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(P=p, R=r, F1=f1, mIoU=float(np.mean(ious)) if ious else 0.0,
                FP_per_img=fp / max(1, len(gt)), sec_per_img=t_total / max(1, len(gt)))


def ablation(img_dir, gt_csv, looks_list=(None, 8, 4, 2)):
    configs = {
        "fixed: none":              Cfg(filter_mode="none", use_shadow=False),
        "fixed: bilateral":         Cfg(filter_mode="bilateral", use_shadow=False),
        "fixed: lee5":              Cfg(filter_mode="lee5", use_shadow=False),
        "fixed: lee9":              Cfg(filter_mode="lee9", use_shadow=False),
        "fixed: nlm":               Cfg(filter_mode="nlm", use_shadow=False),
        "adaptive, no shadow":      Cfg(filter_mode="adaptive", use_shadow=False),
        "adaptive + shadow (ours)": Cfg(filter_mode="adaptive", use_shadow=True),
    }
    rows = []
    for looks in looks_list:
        tag = "original" if looks is None else f"L={looks}"
        for name, cfg in configs.items():
            m = evaluate(img_dir, gt_csv, cfg, looks)
            rows.append((tag, name, m))
            print(f"{tag:9s} {name:26s} P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} "
                  f"mIoU={m['mIoU']:.3f} FP/img={m['FP_per_img']:.2f} {m['sec_per_img']*1000:.0f}ms")
    return rows


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--out", default="result.png")
    ap.add_argument("--img_dir")
    ap.add_argument("--gt")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--axis", default="horizontal", choices=["horizontal", "vertical"])
    a, _ = ap.parse_known_args()

    cfg = Cfg(range_axis=a.axis)
    if a.image:
        matplotlib.use("Agg") if not os.environ.get("DISPLAY") else None
        res = detect(load_gray(a.image), cfg)
        print(f"speckle CV={res['speckle_cv']:.3f}  filter={res['filter_name']}")
        for d in res["dets"]:
            print("box", d["box"], "score %.3f" % d["score"],
                  "shadow %.2f (%s)" % (d["shadow_frac"], d["shadow_side"]),
                  "validated" if d["validated"] else "highlight-only")
        show(res, save=a.out)
    if a.img_dir and a.gt:
        if a.ablation:
            ablation(a.img_dir, a.gt)
        else:
            print(evaluate(a.img_dir, a.gt, cfg))
# =================================================================================== END OF PART P ====================================================================================
