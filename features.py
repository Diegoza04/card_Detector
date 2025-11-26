import cv2
import numpy as np
import math

try:
    import scipy.ndimage as ndi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Aviso: scipy no está instalado. Instala con: pip install scipy para mejor detección de lóbulos de corazones.")

from contours import find_contours

def extract_symbol_color_stats_v3(symbol_binary, corner_rgb, template_name=None):
    contours, _ = find_contours(symbol_binary)
    if not contours:
        return {
            "red_pct_hsv": 0.0, "lab_a_mean": 0.0, "rg_diff_mean": 0.0, "cr_mean": 0.0,
            "lab_a_norm": 0.0, "cr_norm": 0.0, "red_confidence": 0.0, "is_red": False,
            "fallback_color_red": False
        }
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    roi = corner_rgb[y:y+h, x:x+w]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = largest - [x, y]
    cv2.drawContours(mask, [shifted], -1, 255, -1)
    
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    l, a, b_channel = cv2.split(roi_lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_equalized = clahe.apply(l)
    roi_lab_enhanced = cv2.merge([l_equalized, a, b_channel])
    roi_enhanced = cv2.cvtColor(roi_lab_enhanced, cv2.COLOR_LAB2RGB)
    
    hsv = cv2.cvtColor(roi_enhanced, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi_enhanced, cv2.COLOR_RGB2LAB)
    ycrcb = cv2.cvtColor(roi_enhanced, cv2.COLOR_RGB2YCrCb)
    
    R = roi_enhanced[:, :, 0].astype(np.float32)
    G = roi_enhanced[:, :, 1].astype(np.float32)
    B = roi_enhanced[:, :, 2].astype(np.float32)
    denom = (R + G + B + 1e-6)
    rg_diff = (R - G) / denom
    
    symbol_mask = (mask > 0)
    
    lower_sets = [
        ((0, 20, 20), (15, 255, 255)),
        ((5, 20, 20), (25, 255, 255)),
        ((160, 20, 20), (180, 255, 255)),
        ((170, 20, 20), (180, 255, 255))
    ]
    
    red_hsv_mask = np.zeros_like(mask)
    for (l_bound, u_bound) in lower_sets:
        l_arr = np.array(l_bound, dtype=np.uint8)
        u_arr = np.array(u_bound, dtype=np.uint8)
        temp = cv2.inRange(hsv, l_arr, u_arr)
        red_hsv_mask = cv2.bitwise_or(red_hsv_mask, temp)
    
    red_pct_hsv = np.sum((red_hsv_mask > 0) & symbol_mask) / (np.sum(symbol_mask) + 1e-6)
    
    a_lab = lab[:, :, 1]
    Cr = ycrcb[:, :, 1]
    
    lab_a_mean = np.mean(a_lab[symbol_mask]) if np.sum(symbol_mask) > 0 else 0.0
    rg_diff_mean = np.mean(rg_diff[symbol_mask]) if np.sum(symbol_mask) > 0 else 0.0
    cr_mean = np.mean(Cr[symbol_mask]) if np.sum(symbol_mask) > 0 else 0.0
    
    lab_a_norm = (lab_a_mean - 128) / 64.0
    cr_norm = (cr_mean - 128) / 64.0
    
    raw_score = (0.40 * red_pct_hsv + 
                 0.25 * max(0, lab_a_norm) + 
                 0.20 * max(0, rg_diff_mean) + 
                 0.15 * max(0, cr_norm))
    
    red_confidence = 1.0 / (1.0 + math.exp(-10 * (raw_score - 0.15)))
    is_red = red_confidence > 0.25
    
    fallback_color_red = False
    if (template_name in ["Corazones", "Diamantes"]) and not is_red:
        kernel = np.ones((3, 3), np.uint8)
        dil_mask = cv2.dilate(mask, kernel, iterations=2)
        dil_symbol = (dil_mask > 0)
        
        R_mean = np.mean(R[dil_symbol]) if np.sum(dil_symbol) > 0 else 0
        G_mean = np.mean(G[dil_symbol]) if np.sum(dil_symbol) > 0 else 0
        B_mean = np.mean(B[dil_symbol]) if np.sum(dil_symbol) > 0 else 0
        
        if (R_mean - G_mean > 5) and (R_mean - B_mean > 5):
            red_confidence = max(red_confidence, 0.40)
            is_red = True
            fallback_color_red = True
    
    return {
        "red_pct_hsv": red_pct_hsv,
        "lab_a_mean": lab_a_mean,
        "rg_diff_mean": rg_diff_mean,
        "cr_mean": cr_mean,
        "lab_a_norm": lab_a_norm,
        "cr_norm": cr_norm,
        "red_confidence": red_confidence,
        "is_red": is_red,
        "fallback_color_red": fallback_color_red
    }

def compute_shape_metrics(symbol_binary):
    contours,_ = find_contours(symbol_binary)
    if not contours:
        return {
            "area":0,"perimeter":0,"circularity":0,"solidity":0,
            "aspect_ratio":1,"vertices":0,"defects":0,"hu":np.zeros(7)
        }
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4*math.pi*area/(perimeter*perimeter)) if perimeter>0 else 0
    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = h/w if w>0 else 1
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area>0 else 0
    eps = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    vertices = len(approx)
    defects_count = 0
    if len(c)>=3:
        hull_indices = cv2.convexHull(c, returnPoints=False)
        if hull_indices is not None and len(hull_indices)>3:
            defects = cv2.convexityDefects(c, hull_indices)
            if defects is not None:
                for d in defects:
                    depth = d[0][3]/256.0
                    if depth > 4:
                        defects_count += 1
    m = cv2.moments(symbol_binary)
    hu = cv2.HuMoments(m).flatten()
    for i in range(len(hu)):
        if hu[i]!=0:
            hu[i] = -math.copysign(1.0, hu[i]) * math.log10(abs(hu[i]))
    return {
        "area":area,"perimeter":perimeter,"circularity":circularity,
        "solidity":solidity,"aspect_ratio":aspect_ratio,"vertices":vertices,
        "defects":defects_count,"hu":hu
    }

def compute_heart_features(symbol_binary):
    h, w = symbol_binary.shape
    if h < 8 or w < 8:
        return {
            "peak_count": 0,
            "top_bottom_ratio": 1.0,
            "symmetry": 0.0,
            "avg_roundness": 0.0,
            "heart_lobes_score": 0.0
        }
    
    H0, W0 = 64, 64
    norm = cv2.resize(symbol_binary, (W0, W0), interpolation=cv2.INTER_AREA)
    kernel = np.ones((2, 2), np.uint8)
    norm = cv2.morphologyEx(norm, cv2.MORPH_CLOSE, kernel)
    
    top_h = H0 // 3
    top_region = norm[:top_h, :]
    col_sum = top_region.sum(axis=0).astype(np.float32)
    
    if SCIPY_AVAILABLE:
        col_smooth = ndi.gaussian_filter1d(col_sum, sigma=2.5)
    else:
        kernel_smooth = np.ones(7) / 7.0
        col_smooth = np.convolve(col_sum, kernel_smooth, mode='same')
    
    peaks = []
    for i in range(2, len(col_smooth) - 2):
        if (col_smooth[i] > col_smooth[i-1] and col_smooth[i] > col_smooth[i+1] and
            col_smooth[i] > col_smooth[i-2] and col_smooth[i] > col_smooth[i+2]):
            peaks.append(i)
    
    threshold_peak = 0.20 * col_smooth.max() if col_smooth.max() > 0 else 0
    peaks = [p for p in peaks if col_smooth[p] >= threshold_peak]
    
    filtered_peaks = []
    min_distance = W0 // 5
    for peak in peaks:
        if not filtered_peaks or abs(peak - filtered_peaks[-1]) > min_distance:
            filtered_peaks.append(peak)
    
    peaks_count = len(filtered_peaks)
    
    active_cols = np.where(col_smooth > 0.15 * (col_smooth.max() + 1e-6))[0]
    top_width = active_cols[-1] - active_cols[0] + 1 if len(active_cols) > 0 else W0
    
    bottom_region = norm[-top_h:, :]
    bottom_sum = bottom_region.sum(axis=0)
    bottom_active = np.where(bottom_sum > 0.15 * (bottom_sum.max() + 1e-6))[0]
    bottom_width = bottom_active[-1] - bottom_active[0] + 1 if len(bottom_active) > 0 else W0
    top_bottom_ratio = top_width / bottom_width if bottom_width > 0 else 1.0
    
    left_half = top_region[:, :W0 // 2]
    right_half = top_region[:, W0 // 2:]
    right_reflect = np.flip(right_half, axis=1)
    
    left_norm = left_half.astype(np.float32)
    right_norm = right_reflect.astype(np.float32)
    
    if left_norm.max() > 0:
        left_norm = left_norm / left_norm.max()
    if right_norm.max() > 0:
        right_norm = right_norm / right_norm.max()
    
    l_vec = left_norm.flatten()
    r_vec = right_norm.flatten()
    l_vec /= (np.linalg.norm(l_vec) + 1e-6)
    r_vec /= (np.linalg.norm(r_vec) + 1e-6)
    symmetry = float(np.dot(l_vec, r_vec))
    
    gauss_x = np.linspace(-1, 1, W0 // 2)
    gauss_y = np.linspace(-1, 1, top_h)
    gx, gy = np.meshgrid(gauss_x, gauss_y)
    gauss_mask = np.exp(-(gx * gx + gy * gy) * 2.0)
    gauss_mask = (gauss_mask - gauss_mask.min()) / (gauss_mask.max() - gauss_mask.min() + 1e-6)
    
    def roundness_score(lobe):
        if lobe.sum() == 0:
            return 0.0
        ln = lobe.astype(np.float32)
        if ln.max() > 0:
            ln = (ln - ln.min()) / (ln.max() - ln.min() + 1e-6)
        return float(np.sum(ln * gauss_mask) / (np.sum(gauss_mask) + 1e-6))
    
    left_roundness = roundness_score(left_half)
    right_roundness = roundness_score(right_half)
    avg_roundness = (left_roundness + right_roundness) / 2.0
    
    lobes_score = 0.0
    if peaks_count >= 2:
        lobes_score += 0.45
    elif peaks_count == 1:
        lobes_score += 0.10
    
    if top_bottom_ratio > 1.02:
        lobes_score += 0.25
    
    if avg_roundness > 0.40:
        lobes_score += 0.20
    
    if symmetry > 0.80:
        lobes_score += 0.10
    
    lobes_score = min(lobes_score, 1.0)
    
    return {
        "peak_count": peaks_count,
        "top_bottom_ratio": top_bottom_ratio,
        "symmetry": symmetry,
        "avg_roundness": avg_roundness,
        "heart_lobes_score": lobes_score
    }

def compute_clover_features(symbol_binary):
    h, w = symbol_binary.shape
    if h < 8 or w < 8:
        return {
            "lobe_count": 0,
            "base_narrowness": 0.0,
            "top_bottom_complexity": 0.0,
            "convexity_defects_score": 0.0,
            "clover_feature_score": 0.0
        }
    
    H0, W0 = 64, 64
    norm = cv2.resize(symbol_binary, (W0, W0), interpolation=cv2.INTER_AREA)
    kernel = np.ones((2, 2), np.uint8)
    norm = cv2.morphologyEx(norm, cv2.MORPH_CLOSE, kernel)
    
    split_point = int(H0 * 0.65)
    top_region = norm[:split_point, :]
    bottom_region = norm[split_point:, :]
    
    top_projection = top_region.sum(axis=0).astype(np.float32)
    
    if SCIPY_AVAILABLE:
        top_smooth = ndi.gaussian_filter1d(top_projection, sigma=2.0)
    else:
        kernel_smooth = np.ones(5) / 5.0
        top_smooth = np.convolve(top_projection, kernel_smooth, mode='same')
    
    peaks = []
    for i in range(3, len(top_smooth) - 3):
        if (top_smooth[i] > top_smooth[i-1] and top_smooth[i] > top_smooth[i+1] and
            top_smooth[i] > top_smooth[i-2] and top_smooth[i] > top_smooth[i+2] and
            top_smooth[i] > top_smooth[i-3] and top_smooth[i] > top_smooth[i+3]):
            peaks.append(i)
    
    threshold_peak = 0.25 * top_smooth.max() if top_smooth.max() > 0 else 0
    significant_peaks = [p for p in peaks if top_smooth[p] >= threshold_peak]
    
    filtered_peaks = []
    min_distance = W0 // 6
    for peak in significant_peaks:
        if not filtered_peaks or abs(peak - filtered_peaks[-1]) > min_distance:
            filtered_peaks.append(peak)
    
    lobe_count = len(filtered_peaks)
    
    bottom_projection = bottom_region.sum(axis=0).astype(np.float32)
    bottom_active = np.where(bottom_projection > 0.15 * (bottom_projection.max() + 1e-6))[0]
    bottom_width = bottom_active[-1] - bottom_active[0] + 1 if len(bottom_active) > 0 else W0
    
    top_active = np.where(top_smooth > 0.15 * (top_smooth.max() + 1e-6))[0]
    top_width = top_active[-1] - top_active[0] + 1 if len(top_active) > 0 else W0
    
    base_narrowness = 1.0 - (bottom_width / (top_width + 1e-6))
    base_narrowness = max(0, min(1, base_narrowness))
    
    top_variance = np.var(top_smooth)
    bottom_variance = np.var(bottom_projection)
    
    complexity_ratio = top_variance / (bottom_variance + 1e-6)
    top_bottom_complexity = min(complexity_ratio / 3.0, 1.0)
    
    contours, _ = find_contours(norm)
    if contours:
        c = max(contours, key=cv2.contourArea)
        defects_count = 0
        if len(c) >= 5:
            hull_indices = cv2.convexHull(c, returnPoints=False)
            if hull_indices is not None and len(hull_indices) >= 4:
                defects = cv2.convexityDefects(c, hull_indices)
                if defects is not None:
                    for d in defects:
                        depth = d[0][3] / 256.0
                        if depth > 3:
                            defects_count += 1
        
        if defects_count >= 2 and defects_count <= 5:
            convexity_defects_score = 1.0
        elif defects_count == 1 or defects_count == 6:
            convexity_defects_score = 0.6
        else:
            convexity_defects_score = 0.2
    else:
        convexity_defects_score = 0.0
    
    score = 0.0
    if lobe_count == 3:
        score += 0.35
    elif lobe_count == 2:
        score += 0.20
    elif lobe_count == 4:
        score += 0.15
    else:
        score += 0.05
    
    score += 0.25 * base_narrowness
    score += 0.20 * top_bottom_complexity
    score += 0.20 * convexity_defects_score
    
    clover_feature_score = min(score, 1.0)
    
    return {
        "lobe_count": lobe_count,
        "base_narrowness": base_narrowness,
        "top_bottom_complexity": top_bottom_complexity,
        "convexity_defects_score": convexity_defects_score,
        "clover_feature_score": clover_feature_score
    }

def compute_spade_features(symbol_binary):
    h, w = symbol_binary.shape
    if h < 8 or w < 8:
        return {
            "peak_sharpness": 0.0,
            "vertical_aspect": 0.0,
            "lateral_symmetry": 0.0,
            "base_narrowness": 0.0,
            "top_concentration": 0.0,
            "spade_feature_score": 0.0
        }
    
    H0, W0 = 64, 64
    norm = cv2.resize(symbol_binary, (W0, W0), interpolation=cv2.INTER_AREA)
    kernel = np.ones((2, 2), np.uint8)
    norm = cv2.morphologyEx(norm, cv2.MORPH_CLOSE, kernel)
    
    top_third = int(H0 * 0.33)
    top_region = norm[:top_third, :]
    
    top_projection = top_region.sum(axis=1).astype(np.float32)
    
    if top_projection.max() > 0:
        peak_row = np.argmax(top_projection)
        if peak_row < len(top_projection) - 5:
            decay_values = top_projection[peak_row:peak_row+5]
            if len(decay_values) > 1 and decay_values[0] > 0:
                decay_rate = (decay_values[0] - decay_values[-1]) / (decay_values[0] + 1e-6)
                peak_sharpness = min(decay_rate / 0.5, 1.0)
            else:
                peak_sharpness = 0.0
        else:
            peak_sharpness = 0.0
    else:
        peak_sharpness = 0.0
    
    top_horizontal = top_region.sum(axis=0).astype(np.float32)
    
    if SCIPY_AVAILABLE:
        top_h_smooth = ndi.gaussian_filter1d(top_horizontal, sigma=1.5)
    else:
        kernel_smooth = np.ones(3) / 3.0
        top_h_smooth = np.convolve(top_horizontal, kernel_smooth, mode='same')
    
    center_start = W0 // 3
    center_end = 2 * W0 // 3
    center_sum = np.sum(top_h_smooth[center_start:center_end])
    total_sum = np.sum(top_h_smooth) + 1e-6
    top_concentration = center_sum / total_sum
    
    contours, _ = find_contours(norm)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, wc, hc = cv2.boundingRect(c)
        aspect_ratio = hc / (wc + 1e-6)
        if 1.1 <= aspect_ratio <= 1.4:
            vertical_aspect = 1.0
        elif 1.0 <= aspect_ratio <= 1.5:
            vertical_aspect = 0.7
        else:
            vertical_aspect = 0.3
    else:
        vertical_aspect = 0.0
    
    mid_col = W0 // 2
    left_half = norm[:, :mid_col]
    right_half = norm[:, mid_col:]
    right_flipped = np.flip(right_half, axis=1)
    min_width = min(left_half.shape[1], right_flipped.shape[1])
    left_half = left_half[:, :min_width]
    right_flipped = right_flipped[:, :min_width]
    
    left_norm = left_half.astype(np.float32)
    right_norm = right_flipped.astype(np.float32)
    
    if left_norm.max() > 0:
        left_norm = left_norm / left_norm.max()
    if right_norm.max() > 0:
        right_norm = right_norm / right_norm.max()
    
    l_vec = left_norm.flatten()
    r_vec = right_norm.flatten()
    l_vec /= (np.linalg.norm(l_vec) + 1e-6)
    r_vec /= (np.linalg.norm(r_vec) + 1e-6)
    lateral_symmetry = float(np.dot(l_vec, r_vec))
    
    bottom_third = int(H0 * 0.67)
    bottom_region = norm[bottom_third:, :]
    bottom_projection = bottom_region.sum(axis=0).astype(np.float32)
    bottom_active = np.where(bottom_projection > 0.15 * (bottom_projection.max() + 1e-6))[0]
    bottom_width = bottom_active[-1] - bottom_active[0] + 1 if len(bottom_active) > 0 else W0
    
    middle_start = int(H0 * 0.33)
    middle_end = int(H0 * 0.67)
    middle_region = norm[middle_start:middle_end, :]
    middle_projection = middle_region.sum(axis=0).astype(np.float32)
    middle_active = np.where(middle_projection > 0.15 * (middle_projection.max() + 1e-6))[0]
    middle_width = middle_active[-1] - middle_active[0] + 1 if len(middle_active) > 0 else W0
    
    base_narrowness = 1.0 - (bottom_width / (middle_width + 1e-6))
    base_narrowness = max(0, min(1, base_narrowness))
    
    score = 0.0
    score += 0.30 * peak_sharpness
    score += 0.20 * vertical_aspect
    if lateral_symmetry > 0.85:
        score += 0.25
    elif lateral_symmetry > 0.75:
        score += 0.15
    else:
        score += 0.05
    score += 0.15 * base_narrowness
    if top_concentration > 0.45:
        score += 0.10
    elif top_concentration > 0.35:
        score += 0.05
    
    spade_feature_score = min(score, 1.0)
    
    return {
        "peak_sharpness": peak_sharpness,
        "vertical_aspect": vertical_aspect,
        "lateral_symmetry": lateral_symmetry,
        "base_narrowness": base_narrowness,
        "top_concentration": top_concentration,
        "spade_feature_score": spade_feature_score
    }

def compute_diamond_features(symbol_binary):
    h, w = symbol_binary.shape
    if h < 8 or w < 8:
        return {
            "approx_vertices":0,
            "aspect_ratio_ok":False,
            "radial_uniformity":0.0,
            "angle_uniformity":0.0,
            "orientation_score":0.0,
            "diamond_feature_score":0.0
        }
    contours,_ = find_contours(symbol_binary)
    if not contours:
        return {
            "approx_vertices":0,
            "aspect_ratio_ok":False,
            "radial_uniformity":0.0,
            "angle_uniformity":0.0,
            "orientation_score":0.0,
            "diamond_feature_score":0.0
        }
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 30:
        return {
            "approx_vertices":0,
            "aspect_ratio_ok":False,
            "radial_uniformity":0.0,
            "angle_uniformity":0.0,
            "orientation_score":0.0,
            "diamond_feature_score":0.0
        }
    x,y,wc,hc = cv2.boundingRect(c)
    aspect_ratio = hc/wc if wc>0 else 1
    aspect_ratio_ok = 0.8 <= aspect_ratio <= 1.25

    peri = cv2.arcLength(c, True)
    approx_list = []
    for factor in [0.015, 0.02, 0.03]:
        eps = factor * peri
        ap = cv2.approxPolyDP(c, eps, True)
        approx_list.append(ap)
    chosen = min(approx_list, key=lambda ap: abs(len(ap)-4))
    approx_vertices = len(chosen)

    M = cv2.moments(c)
    cx = M["m10"]/M["m00"] if M["m00"]>0 else wc/2
    cy = M["m01"]/M["m00"] if M["m00"]>0 else hc/2
    pts = chosen.reshape(-1,2)
    dists = [math.hypot(px-cx, py-cy) for px,py in pts]
    if dists:
        mean_d = np.mean(dists)
        std_d = np.std(dists)
        radial_uniformity = 1.0 - min(std_d/(mean_d+1e-6), 1.0)
    else:
        radial_uniformity = 0.0

    def angle(a,b,c):
        ab = a-b
        cb = c-b
        dot = ab.dot(cb)
        nab = np.linalg.norm(ab)
        ncb = np.linalg.norm(cb)
        if nab*ncb == 0:
            return 0
        cosang = np.clip(dot/(nab*ncb), -1, 1)
        return math.degrees(math.acos(cosang))
    angles = []
    pts_cycle = np.vstack([pts, pts[0], pts[1]]) if len(pts)>=3 else pts
    for i in range(len(pts)):
        a = pts_cycle[i]
        b = pts_cycle[i+1]
        c2 = pts_cycle[i+2]
        ang = angle(a,b,c2)
        angles.append(ang)
    if angles:
        mean_ang = np.mean(angles)
        std_ang = np.std(angles)
        angle_uniformity = 1.0 - min(std_ang/(mean_ang+1e-6),1.0)
    else:
        angle_uniformity = 0.0

    pts_all = c.reshape(-1,2).astype(np.float32)
    pts_norm = pts_all - np.mean(pts_all, axis=0)
    cov = np.cov(pts_norm.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argmax(eigvals)
    principal = eigvecs[:,idx]
    angle_deg = abs(math.degrees(math.atan2(principal[1], principal[0])))
    dist_to_45 = min(abs(angle_deg-45), abs(angle_deg-135))
    dist_to_0 = min(abs(angle_deg-0), abs(angle_deg-180))
    orientation_score = 1.0 - min(dist_to_45, dist_to_0)/90.0

    score = 0.0
    if approx_vertices == 4:
        score += 0.25
    elif approx_vertices == 5:
        score += 0.18
    elif approx_vertices == 6:
        score += 0.10
    if aspect_ratio_ok:
        score += 0.20
    score += 0.20 * radial_uniformity
    score += 0.20 * angle_uniformity
    score += 0.15 * orientation_score
    diamond_feature_score = min(score, 1.0)

    return {
        "approx_vertices": approx_vertices,
        "aspect_ratio_ok": aspect_ratio_ok,
        "radial_uniformity": radial_uniformity,
        "angle_uniformity": angle_uniformity,
        "orientation_score": orientation_score,
        "diamond_feature_score": diamond_feature_score
    }

def resegment_symbol_if_degenerate(symbol_binary, corner_rgb, template_name=None):
    h,w = symbol_binary.shape
    if h < 10 or w < 10:
        return symbol_binary, False
    contours,_ = find_contours(symbol_binary)
    if not contours:
        return symbol_binary, False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    circularity = (4*math.pi*area/(peri*peri)) if peri>0 else 0
    x,y,wc,hc = cv2.boundingRect(c)
    aspect_ratio = hc / wc if wc>0 else 1
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area>0 else 0
    degenerate = (circularity < 0.05) or (solidity < 0.2) or (aspect_ratio > 3.0)
    if not degenerate:
        return symbol_binary, False

    gray_corner = cv2.cvtColor(corner_rgb, cv2.COLOR_RGB2GRAY)
    thr = cv2.adaptiveThreshold(gray_corner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 5)
    if template_name == "Diamantes":
        kernel = np.ones((3,3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    thr_dil = cv2.dilate(thr, np.ones((3,3), np.uint8), iterations=1)
    contours2,_ = find_contours(thr_dil)
    if not contours2:
        return symbol_binary, False
    h2, w2 = thr_dil.shape
    candidates = []
    for c2 in contours2:
        area2 = cv2.contourArea(c2)
        if area2 < 30:
            continue
        x2,y2,w2c,h2c = cv2.boundingRect(c2)
        if y2 < h2 * 0.85:
            candidates.append((area2, c2))
    if not candidates:
        return symbol_binary, False
    best_c = max(candidates, key=lambda x: x[0])[1]
    x2,y2,w2c,h2c = cv2.boundingRect(best_c)
    new_symbol = thr_dil[y2:y2+h2c, x2:x2+w2c].copy()
    return new_symbol, True

def extract_symbol_color_vector(symbol_binary, corner_rgb):
    contours,_ = find_contours(symbol_binary)
    if not contours:
        return np.array([0,0,0], dtype=np.float32)
    largest = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    roi = corner_rgb[y:y+h, x:x+w]
    mask = np.zeros((h,w), dtype=np.uint8)
    shifted = largest - [x,y]
    cv2.drawContours(mask, [shifted], -1, 255, -1)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    rgb = roi.astype(np.float32)
    R = rgb[:,:,0]; G = rgb[:,:,1]; B = rgb[:,:,2]
    denom = (R+G+B+1e-6)
    rg_diff = (R - G)/denom
    symbol_mask = (mask>0)
    if np.sum(symbol_mask)==0:
        return np.array([0,0,0], dtype=np.float32)
    mean_H = np.mean(hsv[:,:,0][symbol_mask])
    mean_a = np.mean(lab[:,:,1][symbol_mask])
    mean_rg = np.mean(rg_diff[symbol_mask])
    return np.array([mean_H, mean_a, mean_rg], dtype=np.float32)