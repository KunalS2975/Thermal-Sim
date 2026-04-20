"""
Semiconductor Thermal Modeling — Interactive Web App
Flask backend: FDM solver + ML recommender + application predictor
"""

import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MATERIAL DATABASE
# ─────────────────────────────────────────────────────────────────────────────
MATERIALS = {
    "Silicon (Si)": {
        "k": 150, "cost_per_wafer_usd": 15, "cost_tier": "Low",
        "color": "#e63946", "industries": ["Consumer Electronics", "Computing", "Automotive"],
        "notes": "Industry standard. Mature supply chain, lowest cost, good for general-purpose ICs.",
        "max_temp_rating": 150, "density": 2330, "specific_heat": 700
    },
    "Gallium Arsenide (GaAs)": {
        "k": 50, "cost_per_wafer_usd": 250, "cost_tier": "High",
        "color": "#f4a261", "industries": ["Photonics", "RF/Wireless", "Optoelectronics"],
        "notes": "Superior electron mobility. Used in LEDs, solar cells, RF amplifiers. Poor thermal conductor.",
        "max_temp_rating": 350, "density": 5316, "specific_heat": 327
    },
    "Silicon Carbide (SiC)": {
        "k": 490, "cost_per_wafer_usd": 800, "cost_tier": "Very High",
        "color": "#2a9d8f", "industries": ["Power Electronics", "EV/Automotive", "Aerospace", "5G"],
        "notes": "Best thermal conductor. Used in EV inverters, 5G base stations. Premium cost justified by thermal performance.",
        "max_temp_rating": 600, "density": 3210, "specific_heat": 750
    },
    "Gallium Nitride (GaN)": {
        "k": 130, "cost_per_wafer_usd": 500, "cost_tier": "High",
        "color": "#457b9d", "industries": ["RF/Wireless", "Power Electronics", "5G", "Military"],
        "notes": "High breakdown voltage, high frequency. Common in 5G RF front-ends and fast chargers.",
        "max_temp_rating": 400, "density": 6150, "specific_heat": 490
    },
    "Diamond (CVD)": {
        "k": 2200, "cost_per_wafer_usd": 5000, "cost_tier": "Extreme",
        "color": "#9b5de5", "industries": ["Aerospace", "Military", "High-Performance Computing"],
        "notes": "Highest thermal conductivity of any material. Prohibitively expensive for most applications.",
        "max_temp_rating": 700, "density": 3500, "specific_heat": 502
    },
    "Aluminum Nitride (AlN)": {
        "k": 320, "cost_per_wafer_usd": 600, "cost_tier": "Very High",
        "color": "#f4d35e", "industries": ["Power Electronics", "LED Lighting", "Automotive"],
        "notes": "Excellent thermal conductor, electrical insulator. Used as substrate in LED packaging.",
        "max_temp_rating": 500, "density": 3260, "specific_heat": 740
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION DATABASE
# ─────────────────────────────────────────────────────────────────────────────
APPLICATIONS = {
    "EV Inverter": {
        "industry": "Automotive / Power Electronics",
        "typical_Q": 1.5e11, "required_k_min": 300, "priority": "thermal",
        "budget": "high", "operating_temp_max": 175,
        "description": "Converts DC battery power to AC for motor drive. Extreme heat density, must survive vibration.",
        "recommended_materials": ["Silicon Carbide (SiC)", "Aluminum Nitride (AlN)"]
    },
    "5G Base Station PA": {
        "industry": "Telecommunications",
        "typical_Q": 8e10, "required_k_min": 100, "priority": "performance",
        "budget": "high", "operating_temp_max": 150,
        "description": "Power amplifier in 5G mmWave base station. High frequency, moderate heat density.",
        "recommended_materials": ["Gallium Nitride (GaN)", "Silicon Carbide (SiC)"]
    },
    "CPU / Microprocessor": {
        "industry": "Computing",
        "typical_Q": 1e11, "required_k_min": 100, "priority": "balanced",
        "budget": "medium", "operating_temp_max": 100,
        "description": "General-purpose processor. Balance of performance, cost, and thermal management.",
        "recommended_materials": ["Silicon (Si)", "Silicon Carbide (SiC)"]
    },
    "LED Array": {
        "industry": "Lighting / Photonics",
        "typical_Q": 4e10, "required_k_min": 150, "priority": "balanced",
        "budget": "medium", "operating_temp_max": 125,
        "description": "High-power LED module. Thermal management critical for lumen output and lifetime.",
        "recommended_materials": ["Aluminum Nitride (AlN)", "Silicon Carbide (SiC)"]
    },
    "RF Amplifier (Satellite)": {
        "industry": "Aerospace / Defense",
        "typical_Q": 2e11, "required_k_min": 200, "priority": "performance",
        "budget": "extreme", "operating_temp_max": 200,
        "description": "High-power amplifier for satellite communication. Budget is no constraint; reliability is paramount.",
        "recommended_materials": ["Diamond (CVD)", "Silicon Carbide (SiC)"]
    },
    "Solar Cell": {
        "industry": "Renewable Energy",
        "typical_Q": 2e10, "required_k_min": 40, "priority": "cost",
        "budget": "low", "operating_temp_max": 85,
        "description": "Photovoltaic cell. Cost sensitivity is high; moderate thermal requirements.",
        "recommended_materials": ["Silicon (Si)", "Gallium Arsenide (GaAs)"]
    },
    "LIDAR Module (Autonomous Vehicle)": {
        "industry": "Automotive / Sensors",
        "typical_Q": 6e10, "required_k_min": 80, "priority": "balanced",
        "budget": "medium", "operating_temp_max": 125,
        "description": "Laser ranging system. Moderate power, compact form factor, automotive-grade reliability.",
        "recommended_materials": ["Gallium Arsenide (GaAs)", "Silicon (Si)"]
    },
    "Power MOSFET (Fast Charger)": {
        "industry": "Consumer Electronics / Power",
        "typical_Q": 9e10, "required_k_min": 100, "priority": "balanced",
        "budget": "medium", "operating_temp_max": 150,
        "description": "Switching transistor in USB-C/GaN fast chargers. High switching frequency, compact size.",
        "recommended_materials": ["Gallium Nitride (GaN)", "Silicon (Si)"]
    },
    "Radar Transmitter (Defense)": {
        "industry": "Defense / Aerospace",
        "typical_Q": 3e11, "required_k_min": 400, "priority": "thermal",
        "budget": "extreme", "operating_temp_max": 300,
        "description": "Phased array radar. Very high power density, extreme reliability requirement.",
        "recommended_materials": ["Diamond (CVD)", "Silicon Carbide (SiC)"]
    },
    "Laser Diode": {
        "industry": "Photonics / Medical",
        "typical_Q": 1.2e11, "required_k_min": 60, "priority": "performance",
        "budget": "high", "operating_temp_max": 80,
        "description": "Semiconductor laser for fiber optics or medical imaging. Temperature-sensitive wavelength.",
        "recommended_materials": ["Gallium Arsenide (GaAs)", "Gallium Nitride (GaN)"]
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FDM SOLVER
# Domain: 2 mm × 2 mm chip-scale (compact, physically meaningful)
# Q range shown to user: 1e9–5e11 W/m³ (realistic microelectronics power density)
# ─────────────────────────────────────────────────────────────────────────────
NX = 60
T_BC = 300.0
CHIP_XF = (0.35, 0.65)   # chip spans central 30% in x
CHIP_YF = (0.00, 0.25)   # chip spans top 25% in y
DOMAIN   = 2e-3           # 2 mm compact domain

def solve_fdm(nx, ny, k_chip, Q, T_bc=T_BC, max_iter=10000, tol=1e-4):
    dx     = DOMAIN / (nx - 1)
    chip_rows = max(1, int(ny * CHIP_YF[1]))
    ccs    = int(nx * CHIP_XF[0])
    cce    = int(nx * CHIP_XF[1])
    k_sub  = k_chip * 0.3

    T      = np.full((ny, nx), T_bc)
    k_map  = np.full((ny, nx), k_sub)
    q_map  = np.zeros((ny, nx))
    k_map[:chip_rows, ccs:cce] = k_chip
    q_map[:chip_rows, ccs:cce] = Q

    coeff = 4.0 / dx ** 2
    for _ in range(max_iter):
        T_old = T.copy()
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                k = k_map[j, i]; q = q_map[j, i]
                T[j, i] = ((T[j,i+1]+T[j,i-1]+T[j+1,i]+T[j-1,i]) / dx**2 + q/k) / coeff
        if np.max(np.abs(T - T_old)) < tol:
            break

    return {
        "T": T.tolist(),
        "Tmax":  round(float(T.max()), 3),
        "Tmean": round(float(T.mean()), 3),
        "Trise": round(float(T.max() - T_bc), 3),
        "chip_rows": chip_rows,
        "ccs": ccs,
        "cce": cce,
        "nx": nx,
        "ny": ny,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL — train on synthetic parametric data
# ─────────────────────────────────────────────────────────────────────────────
def build_ml_model():
    """Surrogate model: (k, Q, ny) → Tmax trained on FDM-consistent physics."""
    rng = np.random.default_rng(42)
    rows = []
    k_vals  = np.linspace(30, 2500, 20)
    q_vals  = np.linspace(1e9,  5e11, 20)
    ny_vals = [40, 60, 80, 100, 120]
    for k in k_vals:
        for Q in q_vals:
            for ny in ny_vals:
                L_chip  = 0.6e-3          # chip half-width in compact domain
                Tmax_approx = T_BC + (Q * L_chip**2) / (2 * k) * (1 + 0.12 * (80/ny))
                noise = rng.normal(0, 0.5)
                rows.append([k, Q, ny, Tmax_approx + noise])
    data = np.array(rows)
    X, y = data[:, :3], data[:, 3]
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                       learning_rate=0.08, random_state=42)
    model.fit(X, y)
    return model

ML_MODEL = build_ml_model()

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────
def score_material_for_app(mat_name, mat_props, app_props):
    """Score 0-100 for how well a material suits an application."""
    score = 0
    k = mat_props["k"]
    cost = mat_props["cost_per_wafer_usd"]
    max_t = mat_props["max_temp_rating"]
    req_k = app_props["required_k_min"]
    budget = app_props["budget"]
    priority = app_props["priority"]
    op_temp = app_props["operating_temp_max"]

    # Thermal adequacy (0-40)
    if k >= req_k:
        thermal_score = min(40, 25 + 15 * min(1, (k - req_k) / (req_k + 1)))
    else:
        thermal_score = max(0, 40 * (k / req_k) ** 2)
    score += thermal_score

    # Temperature rating (0-20)
    if max_t >= op_temp:
        score += 20
    else:
        score += max(0, 20 * (max_t / op_temp))

    # Cost fit (0-25)
    budget_map = {"low": 20, "medium": 100, "high": 600, "extreme": 10000}
    budget_limit = budget_map.get(budget, 200)
    if cost <= budget_limit:
        score += 25
    else:
        score += max(0, 25 * (budget_limit / cost))

    # Priority alignment (0-15)
    if priority == "thermal" and k >= req_k * 1.5:
        score += 15
    elif priority == "cost" and cost <= 50:
        score += 15
    elif priority == "performance" and k >= req_k:
        score += 12
    elif priority == "balanced":
        score += 10

    return round(min(100, score), 1)

def recommend_for_application(app_name_or_desc):
    """Given an application name (or partial), return recommendations."""
    # Match known apps
    matched_app = None
    for name, props in APPLICATIONS.items():
        if app_name_or_desc.lower() in name.lower() or name.lower() in app_name_or_desc.lower():
            matched_app = (name, props)
            break

    if not matched_app:
        # Fuzzy: pick closest by keyword overlap
        keywords = set(app_name_or_desc.lower().split())
        best_score = 0
        for name, props in APPLICATIONS.items():
            words = set((name + " " + props["industry"] + " " + props["description"]).lower().split())
            overlap = len(keywords & words)
            if overlap > best_score:
                best_score = overlap
                matched_app = (name, props)

    app_name, app_props = matched_app
    scores = {}
    for mat_name, mat_props in MATERIALS.items():
        scores[mat_name] = score_material_for_app(mat_name, mat_props, app_props)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    pred_Tmax = {}
    for mat_name, mat_props in MATERIALS.items():
        pred_Tmax[mat_name] = round(float(ML_MODEL.predict(
            [[mat_props["k"], app_props["typical_Q"], 80]])[0]), 2)

    return {
        "matched_app": app_name,
        "app_props": {
            "industry": app_props["industry"],
            "typical_Q": app_props["typical_Q"],
            "required_k_min": app_props["required_k_min"],
            "priority": app_props["priority"],
            "budget": app_props["budget"],
            "description": app_props["description"],
            "operating_temp_max": app_props["operating_temp_max"],
        },
        "ranked_materials": [
            {
                "name": m,
                "score": s,
                "k": MATERIALS[m]["k"],
                "cost_tier": MATERIALS[m]["cost_tier"],
                "cost_usd": MATERIALS[m]["cost_per_wafer_usd"],
                "color": MATERIALS[m]["color"],
                "notes": MATERIALS[m]["notes"],
                "pred_Tmax": pred_Tmax[m],
                "industries": MATERIALS[m]["industries"],
                "is_recommended": m in app_props["recommended_materials"],
            }
            for m, s in ranked
        ],
        "optimal_k": ranked[0][1],
        "optimal_material": ranked[0][0],
    }

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/solve", methods=["POST"])
def api_solve():
    data = request.get_json()
    k    = float(data.get("k", 150))
    Q    = float(data.get("Q", 1e7))
    ny   = int(data.get("ny", 80))
    k = max(10, min(k, 3000))
    Q = max(1e5, min(Q, 5e7))
    ny = max(30, min(ny, 150))
    result = solve_fdm(NX, ny, k, Q)
    result["pred_Tmax"] = round(float(ML_MODEL.predict([[k, Q, ny]])[0]), 3)
    return jsonify(result)

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()
    app_input = data.get("application", "")
    if not app_input.strip():
        return jsonify({"error": "Please enter an application name."}), 400
    result = recommend_for_application(app_input)
    return jsonify(result)

@app.route("/api/materials", methods=["GET"])
def api_materials():
    return jsonify({
        name: {**props, "T": []} for name, props in MATERIALS.items()
    })

@app.route("/api/applications", methods=["GET"])
def api_applications():
    return jsonify(list(APPLICATIONS.keys()))

@app.route("/api/predict_tmax", methods=["POST"])
def api_predict():
    data = request.get_json()
    k  = float(data.get("k", 150))
    Q  = float(data.get("Q", 1e7))
    ny = float(data.get("ny", 80))
    pred = float(ML_MODEL.predict([[k, Q, ny]])[0])
    return jsonify({"predicted_Tmax": round(pred, 3)})

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Semiconductor Thermal Modeling App")
    print("  Open your browser at: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
