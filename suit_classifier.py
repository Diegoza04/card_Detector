import numpy as np
from symbols import enhanced_match_symbol_v2
from features import (
    extract_symbol_color_stats_v3,
    compute_shape_metrics,
    compute_heart_features,
    compute_diamond_features,
    compute_clover_features,
    compute_spade_features,
    resegment_symbol_if_degenerate,
    extract_symbol_color_vector
)

def classify_suit_v7(suit_symbol_binary, corner_rgb, suit_templates, suit_color_prototypes):
    best_name_template, template_score, template_detail = enhanced_match_symbol_v2(
        suit_symbol_binary, suit_templates, "suit"
    )

    color_stats = extract_symbol_color_stats_v3(suit_symbol_binary, corner_rgb, template_name=best_name_template)
    shape = compute_shape_metrics(suit_symbol_binary)
    heart_feats = compute_heart_features(suit_symbol_binary)
    diamond_feats = compute_diamond_features(suit_symbol_binary)
    clover_feats = compute_clover_features(suit_symbol_binary)
    spade_feats = compute_spade_features(suit_symbol_binary)

    degenerate_fix_applied = False
    if (shape["circularity"] < 0.05) or (shape["solidity"] < 0.2) or (shape["aspect_ratio"] > 3.0):
        new_symbol, applied = resegment_symbol_if_degenerate(suit_symbol_binary, corner_rgb, template_name=best_name_template)
        if applied:
            degenerate_fix_applied = True
            suit_symbol_binary = new_symbol
            color_stats = extract_symbol_color_stats_v3(suit_symbol_binary, corner_rgb, template_name=best_name_template)
            shape = compute_shape_metrics(suit_symbol_binary)
            heart_feats = compute_heart_features(suit_symbol_binary)
            diamond_feats = compute_diamond_features(suit_symbol_binary)
            clover_feats = compute_clover_features(suit_symbol_binary)
            spade_feats = compute_spade_features(suit_symbol_binary)

    red_flag = color_stats["is_red"]
    color_group = "red" if red_flag else "black"

    candidate_suits = ["Diamantes", "Corazones"] if color_group == "red" else ["Picas", "Treboles"]

    proto_dist = {}
    if color_group == "red" and ("Corazones" in suit_color_prototypes) and ("Diamantes" in suit_color_prototypes):
        vec_current = extract_symbol_color_vector(suit_symbol_binary, corner_rgb)
        for s in ["Corazones","Diamantes"]:
            proto = suit_color_prototypes.get(s)
            if proto is not None:
                proto_dist[s] = np.linalg.norm(vec_current - proto)
        if proto_dist:
            max_d = max(proto_dist.values())
            min_d = min(proto_dist.values())
            if max_d - min_d < 1e-6:
                for k in proto_dist:
                    proto_dist[k] = 0.5
            else:
                for k in proto_dist:
                    proto_dist[k] = 1 - (proto_dist[k] - min_d)/(max_d - min_d)

    per_suit_scores = {}
    strong_template_diamond = (best_name_template == "Diamantes" and template_score >= 0.88)
    strong_template_heart   = (best_name_template == "Corazones" and template_score >= 0.88)
    strong_template_clover  = (best_name_template == "Treboles" and template_score >= 0.85)
    strong_template_spade   = (best_name_template == "Picas" and template_score >= 0.85)

    for suit in candidate_suits:
        base = template_score if best_name_template == suit else template_score * 0.85
        heur = 0.0

        if color_group == "red":
            if suit == "Diamantes":
                df = diamond_feats["diamond_feature_score"]
                heur += df * 0.70
                if diamond_feats["approx_vertices"] in [4,5]: 
                    heur += 0.08
                if diamond_feats["aspect_ratio_ok"]: 
                    heur += 0.05
                if diamond_feats["radial_uniformity"] > 0.6: 
                    heur += 0.04
                if diamond_feats["angle_uniformity"] > 0.6: 
                    heur += 0.04
                if diamond_feats["orientation_score"] > 0.5: 
                    heur += 0.04
                
                if heart_feats["heart_lobes_score"] > 0.55:
                    heur -= 0.10
                
                if suit in proto_dist:
                    heur += 0.10 * proto_dist[suit]
                if strong_template_diamond:
                    heur += 0.06
                    
            elif suit == "Corazones":
                lobes = heart_feats["heart_lobes_score"]
                heur += lobes * 0.80
                if heart_feats["peak_count"] >= 2: 
                    heur += 0.10
                if heart_feats["top_bottom_ratio"] > 1.05: 
                    heur += 0.06
                if heart_feats["symmetry"] > 0.85: 
                    heur += 0.06
                
                if diamond_feats["diamond_feature_score"] > 0.50:
                    heur -= 0.08
                
                if suit in proto_dist:
                    heur += 0.12 * proto_dist[suit]
                if strong_template_heart:
                    heur += 0.06
                    
        else:
            if suit == "Picas":
                sp = spade_feats["spade_feature_score"]
                heur += sp * 0.75
                if spade_feats["peak_sharpness"] > 0.5:
                    heur += 0.10
                if spade_feats["vertical_aspect"] > 0.5:
                    heur += 0.08
                if spade_feats["lateral_symmetry"] > 0.8:
                    heur += 0.06
                if spade_feats["base_narrowness"] > 0.3:
                    heur += 0.05
                
                if clover_feats["clover_feature_score"] > 0.45:
                    heur -= 0.15
                if clover_feats["lobe_count"] == 3:
                    heur -= 0.10
                if clover_feats["base_narrowness"] > 0.3:
                    heur -= 0.08
                
                if strong_template_spade:
                    heur += 0.08
                    
            elif suit == "Treboles":
                cf = clover_feats["clover_feature_score"]
                heur += cf * 0.75
                if clover_feats["lobe_count"] == 3:
                    heur += 0.12
                elif clover_feats["lobe_count"] == 2:
                    heur += 0.06
                if clover_feats["base_narrowness"] > 0.25:
                    heur += 0.08
                if clover_feats["top_bottom_complexity"] > 0.4:
                    heur += 0.06
                if clover_feats["convexity_defects_score"] > 0.6:
                    heur += 0.06
                if 0.85 <= shape["aspect_ratio"] <= 1.15:
                    heur += 0.08
                if shape["defects"] >= 2:
                    heur += 0.10
                if shape["circularity"] >= 0.60:
                    heur += 0.06
                if shape["solidity"] >= 0.85:
                    heur += 0.05
                if diamond_feats["diamond_feature_score"] > 0.50:
                    heur -= 0.15
                if diamond_feats["approx_vertices"] == 4:
                    heur -= 0.10
                if diamond_feats["radial_uniformity"] > 0.7:
                    heur -= 0.08
                if strong_template_clover:
                    heur += 0.10

        if degenerate_fix_applied:
            if strong_template_diamond and suit == "Diamantes":
                heur += 0.10
                base += 0.05
            elif strong_template_clover and suit == "Treboles":
                heur += 0.12
                base += 0.06

        final_score = base + heur
        per_suit_scores[suit] = {"base": base, "heur": heur, "final": final_score}

    chosen = max(per_suit_scores.items(), key=lambda x: x[1]["final"])[0]
    final_score = per_suit_scores[chosen]["final"]
    if final_score > 1.0: 
        final_score = 1.0

    debug_info = {
        "template_name": best_name_template,
        "template_score": template_score,
        "template_detail": template_detail,
        "color_stats": color_stats,
        "color_group": color_group,
        "shape": shape,
        "heart_features": heart_feats,
        "diamond_features": diamond_feats,
        "clover_features": clover_feats,
        "spade_features": spade_feats,
        "per_suit_scores": per_suit_scores,
        "proto_similarity": proto_dist,
        "degenerate_fix_applied": degenerate_fix_applied
    }

    if strong_template_diamond and diamond_feats["diamond_feature_score"] >= 0.45 and chosen != "Diamantes":
        if heart_feats["heart_lobes_score"] < 0.70:
            debug_info["override"] = "Forzado Diamantes por template + diamond_feature_score"
            chosen = "Diamantes"
            final_score = max(final_score, 0.80)
    
    if color_group == "black":
        if strong_template_clover and clover_feats["clover_feature_score"] >= 0.40 and chosen != "Treboles":
            if diamond_feats["diamond_feature_score"] < 0.60:
                debug_info["override"] = "Forzado Treboles por template + clover_feature_score"
                chosen = "Treboles"
                final_score = max(final_score, 0.75)
        
        if clover_feats["clover_feature_score"] >= 0.65 and clover_feats["lobe_count"] == 3:
            if chosen != "Treboles":
                debug_info["override"] = "Forzado Treboles por características muy fuertes (3 lóbulos)"
                chosen = "Treboles"
                final_score = max(final_score, 0.80)

    return chosen, final_score, debug_info