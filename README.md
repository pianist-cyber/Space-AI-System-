# Space AI System

An AI-powered scientific computing project that analyzes exoplanet data using Machine Learning and predicts planetary characteristics such as habitability and planet type.

This project combines:

* Astronomy
* Artificial Intelligence
* Data Science
* Scientific Visualization
* Research-Oriented Computing

# Project Overview

The Space AI System is a collaborative Python project built to explore how Machine Learning can be used in astronomy and exoplanet analysis.

The system analyzes planetary data such as:

* mass
* radius
* temperature
* gravity
* orbital distance
* atmospheric conditions

Using multiple KNN (K-Nearest Neighbors) models, the project predicts:

1. Whether a planet may be habitable
2. What category/type of planet it belongs to

# Machine Learning Models

🔹 Habitability Predictor

Predicts:

* Habitable
* Not Habitable

This model compares planetary properties with previously classified planets.

---

🔹 Planet Type Classifier

Predicts:

* Rocky Planet
* Gas Giant
* Ice Giant
* Earth-like Planet

This helps classify planets based on structural and environmental similarities.

# Project Structure

```bash
space_ai_system/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── models/
│   ├── habitability_knn.py
│   ├── planet_type_knn.py
│   └── train_models.py
│
├── preprocessing/
│   ├── cleaner.py
│   ├── scaler.py
│   └── feature_engineering.py
│
├── visualization/
│   ├── graphs.py
│   ├── plots_3d.py
│   └── comparison_charts.py
│
├── reports/
│   └── report_generator.py
│
├── utils/
│   ├── config.py
│   ├── helpers.py
│   └── logger.py
│
├── notebooks/
│
├── tests/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

# Technologies & Libraries

## Core Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn

## Scientific Computing

* Astropy

## Visualization

* Plotly

# Workflow

1. Data Collection

Load planetary datasets from CSV files.

2. Data Preprocessing

* cleaning
* handling missing values
* feature scaling
* feature engineering

3. Model Training

Train multiple KNN models using planetary features.

4. Prediction System

Generate:

* habitability predictions
* planet classification predictions

5. Visualization

Create scientific graphs and comparison charts.

6. Report Generation

Generate summarized planet analysis reports.

# Collaboration

This project is designed for collaborative GitHub development.

Possible team division:

* AI & preprocessing systems
* visualization & reporting systems

GitHub is used for:

* version control
* collaboration
* feature management
* project organization

# Project Goals

The goal of this project is to learn:

* Machine Learning fundamentals
* KNN algorithm implementation
* Data preprocessing pipelines
* Scientific computing
* Visualization systems
* Project architecture
* GitHub collaboration workflows

# Future Improvements

Planned future expansions:

* NASA API integration
* Interactive dashboards
* 3D space visualization
* Additional ML algorithms
* Habitability scoring system
* Real-time planetary analysis

# Dataset

The project uses exoplanet and planetary datasets containing:

* mass
* radius
* temperature
* gravity
* orbital properties
* atmospheric indicators

Datasets may include:

* NASA Exoplanet Archive
* Kaggle exoplanet datasets

# Installation

Clone the repository:

```bash
git clone <repository-link>
cd space_ai_system
```

Install required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```
# Status

Currently in active development.
