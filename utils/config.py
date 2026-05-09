import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths

PATHS = {
    "raw_data"    : os.path.join(BASE_DIR, "data", "raw", "exoplanet_dataset.csv"),
    "cleaned_data": os.path.join(BASE_DIR, "data", "processed", "cleaned_planet_data.csv"),
    "output"      : os.path.join(BASE_DIR, "data", "outputs", "prediction_results.csv"),
    "models_dir"  : os.path.join(BASE_DIR, "models"),
    "reports_dir" : os.path.join(BASE_DIR, "reports"),
    "logs_dir"    : os.path.join(BASE_DIR, "logs"),
}

# KNN Model Parameters

KNN = {
    "habitability_k"        : 5,
    "planet_type_k"         : 7,
    "metric"                : "euclidean",
    "weights"               : "uniform",
    "cross_validation_folds": 5,
    "test_size"             : 0.2,
    "random_state"          : 42,
}

# Dataset Column Names

COLUMNS = {
    "planet_radius"    : "koi_prad",
    "temperature"      : "koi_teq",
    "stellar_flux"     : "koi_insol",
    "orbital_period"   : "koi_period",
    "planet_mass"      : "koi_bmassj",
    "surface_gravity"  : "koi_slogg",
    "stellar_radius"   : "koi_srad",
    "semi_major_axis"  : "koi_dor",
    "eccentricity"     : "koi_eccen",
    "disposition"      : "koi_disposition",
}

# Features Per Model

FEATURES = {
    "habitability": [
        "koi_prad",
        "koi_teq",
        "koi_insol",
        "koi_period",
        "koi_dor",
    ],
    "planet_type": [
        "koi_prad",
        "koi_bmassj",
        "koi_teq",
        "koi_slogg",
        "koi_srad",
    ],
}

# Label Definitions

LABELS = {
    "habitability": {
        "habitable"    : 1,
        "not_habitable": 0,
    },
    "planet_types": ["Rocky", "Gas Giant", "Ice Giant", "Earth-like"],
}

# Habitability Thresholds (derived from known science) 
# Used in feature_engineering.py to generate habitable labels

HABITABILITY_THRESHOLDS = {
    "min_temperature"  : 180,    # Kelvin
    "max_temperature"  : 310,    # Kelvin
    "min_stellar_flux" : 0.25,   # relative to Earth (S_earth)
    "max_stellar_flux" : 1.75,
    "min_radius"       : 0.5,    # relative to Earth radius
    "max_radius"       : 2.0,
}

# Planet Type Thresholds 
# Used in feature_engineering.py to generate planet_type labels

PLANET_TYPE_THRESHOLDS = {
    "rocky_max_radius"    : 1.5,    # Earth radii
    "earth_like_max_radius": 2.0,
    "ice_giant_max_radius" : 6.0,
    # anything above ice_giant_max_radius → Gas Giant
}

# Preprocessing Settings 

PREPROCESSING = {
    "missing_value_strategy": "median",     # median | mean | drop
    "scaling_method"        : "standard",   # standard | minmax | robust
    "outlier_std_threshold" : 3.0,
}

# Visualization Settings 

VISUALIZATION = {
    "figure_size"   : (12, 8),
    "color_palette" : "coolwarm",
    "dpi"           : 150,
    "3d_marker_size": 5,
    "save_figures"  : True,
    "figures_dir"   : os.path.join(BASE_DIR, "data", "outputs", "figures"),
}

# Report Settings 

REPORT = {
    "title"  : "Space AI System — Exoplanet Analysis Report",
    "author" : "Space AI Team",
    "format" : "pdf",
}

# Logging 

LOGGING = {
    "level"  : "INFO",
    "format" : "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    "logfile": os.path.join(BASE_DIR, "logs", "space_ai.log"),
}