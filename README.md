# Mental Health in Tech Analysis

## Overview
This project analyzes mental health data in the technology industry using clustering algorithms to identify patterns and relationships between various factors including workplace conditions, employee demographics, and mental health experiences.

## Features
- Data cleaning and preprocessing of mental health survey responses
- Cluster analysis using KModes algorithm
- Comprehensive visualization of cluster characteristics
- Statistical analysis including Chi-square tests and Cramer's V calculations
- Multiple Correspondence Analysis (MCA) for dimensionality reduction

## Requirements
```
prince==0.13.1
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.1
scipy==1.14.0
kmodes==0.12.2
```

## Project Structure
- `mental_health_clean.py`: Data preprocessing and cleaning operations
- `mental_health_model.py`: Implementation of clustering models and encoders
- `mental_health_analysis.py`: Statistical analysis and feature importance calculations
- `mental_health_viz.py`: Visualization scripts for cluster analysis results

## Installation
1. Clone the repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Processing
The project processes several key aspects of mental health data:
- Demographics (age, gender, country)
- Company information (size, benefits)
- Mental health history (diagnosed conditions, family history)
- Workplace attitudes (disclosure comfort, treatment seeking)
- Job roles and positions

## Analysis Features
- **Clustering**: Uses KModes algorithm to group similar responses
- **Feature Importance**: Calculates Chi-square statistics and Cramer's V coefficients
- **Visualization**: Generates multiple plots

## Visualizations
The project generates several visualization types:
- Pie charts for categorical distributions
- Bar charts for multi-category comparisons
- MCA plots for cluster visualization
- Heatmaps for correlation analysis
