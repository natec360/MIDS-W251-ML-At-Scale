# W251: Predicting Flight Delays through Machine Learning Classifiers at Scale

## Overview
This project uses machine learning models at scale to predict flight delays using a rich set of features, including weather data, prior aircraft delays, and network delay patterns. The goal was to build models that could accurately identify delayed flights in advance, minimizing costs and improving airline operations.

## Team
**W261 Fall 2022, Section 5, Group 4**  
Nathan Chiu, Dominic Lim, Raul Merino, Javier Rondon

## Motivation
False negatives (i.e., predicting a flight will be on time when it is delayed) are costly. Thus, we focused on **maximizing recall**, using the **F2 score** as our primary metric, which weights recall more heavily than precision.

## Dataset
- **Flight data**: 41 million rows from 6 years, filtered to include only US flights with 54 engineered features.
- **Weather data**: 31 million rows from 379 US weather stations, joined to both origin and destination airports.
- Joined via composite key of `(airport, timestamp)` rounded to the hour.

## Data Engineering Pipeline
1. **Join Raw Files**  
   - Airport codes, timezones, weather stations
   - Standardized timestamps (UTC), removed duplicates

2. **Feature Engineering**  
   - **Previous Flight Delay**: Tracks aircraft delay history using tail number
   - **Weather Indicators**: Fog, ice, snow, thunderstorms, etc.
   - **Pagerank**: Measures airport network influence
   - **Delay States**: Clusters of delay behavior at given timestamps
   - **Airport Capacity**: Ratio of actual vs scheduled departures

3. **Model Dataset Prep**
   - Final joined dataset split using **Blocked Time Series Cross Validation**
   - Exported to parquet files

## Modeling Approach

### Models Used
- **Logistic Regression** (initial model)
- **Decision Tree**
- **Random Forest**
- **MLP (Multi-Layer Perceptron)**
- **Ensemble Voting Model**: Voting mechanisms included one-positive, one-negative, and majority voting

### Feature Importance
Feature categories with highest importance:
- `PREV_DEP_DELAY` (Previous Flight Delay)
- `PER_DELAY_15_ORIGIN_LAST_3` (Recent origin delay ratio)
- Pagerank, Weather, and Delay States

### Evaluation Metrics
- **F2 Score**: Emphasizes recall to reduce false negatives
- **Precision**
- **Recall**

## Results

### Best Model: Ensemble Voting (One Positive)
- **F2 Score**: 0.558
- **Precision**: 0.366
- **Recall**: 0.643

### Logistic Regression Baseline
- **Precision**: 0.8208
- **Recall**: 1.0000
- **F2 Score**: 0.9582  
(Note: Skewed due to imbalance and predicting all flights as delayed)

## Key Insights
- **Weather** is responsible for ~20% of total delay minutes.
- **Delays increase throughout the day**, likely due to network effects.
- **Previous aircraft delay** shows strong correlation with current delay.
- **Pagerank & Delay State** features help capture network-based delay propagation.

## Wins
- Developed and tested **novel ensemble voting mechanisms**
- Built and evaluated **feature-rich models using Spark MLlib**
