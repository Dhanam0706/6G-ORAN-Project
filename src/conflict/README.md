
# Explainable ML-Based Conflict Detection in 6G/O-RAN Networks

## 1. Project Overview

This project detects potentially conflicting control actions issued by different xApps in O-RAN networks. It retrieves control actions and RAN measurements from PostgreSQL, identifies action interactions, aligns actions with network measurements, and extracts features for conflict prediction.

An XGBoost classifier predicts conflict probability, while SHAP explains the features influencing the predictions. The system focuses on conflict detection and monitoring, not automatic conflict resolution or retraining.

## 2. Technologies Used

- Python
- PostgreSQL
- Pandas and NumPy
- XGBoost and Scikit-learn
- SHAP
- psycopg2 and python-dotenv

## 3. File Descriptions

### Source Code

- `src/db_reader.py`: Retrieves required records from the database.
- `src/live_db_reader.py`: Supports live and incremental retrieval of control-action and network data.
- `src/state_aligner.py`: Aligns control actions with relevant network measurements using timestamps and identifiers.
- `src/autonomous_monitor.py`: Coordinates data retrieval, conflict prediction, and monitoring output updates.

### Data Files

- `data/raw/xapp_network_data.csv`: Synthetic input data used by the prototype.
- `data/processed/interactions.csv`: Intermediate data representing identified or prepared action interactions.
- `data/processed/conflict_features.csv`: Engineered feature dataset used for model training or evaluation.
- `data/monitor/monitor_results.csv`: Stores generated monitoring and prediction results.
- `data/monitor/monitor_state.json`: Stores the processing checkpoint used by the monitoring workflow.

### Model Files

- `models/conflict_xgboost.pkl`: Trained XGBoost model for the prototype workflow.
- `models/conflict_xgboost_stratified.pkl`: Trained XGBoost model used by the live conflict-detection workflow.

### Configuration Files

- `requirements.txt`: Lists the Python dependencies required to run the project.
- `.env.example`: Provides a template for database connection settings. Create a local `.env` file with valid credentials. Never upload passwords or database secrets to GitHub.

Note: The file descriptions above apply only if the corresponding files are included in the submitted project.

## 4. Implementation Procedure

1. Clone the repository and navigate to the project directory.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

   `pip install -r requirements.txt`

4. Configure the PostgreSQL connection details in the `.env` file.
5. Ensure the required database tables, views, and trained model are accessible.
6. Run the live monitoring workflow from the project root:

   `python src/autonomous_monitor.py`

7. Inspect `data/monitor/monitor_results.csv` for prediction results and `data/monitor/monitor_state.json` for the processing checkpoint.

The live workflow requires authorized access to the configured PostgreSQL database. Prototype preprocessing and training commands depend on the scripts included in the repository.

## 5. Results

The live model was evaluated on 1,242 test records and achieved:

- ROC-AUC: 0.9322
- PR-AUC: 0.4101
- Recall: 91.09%
- Precision: 37.10%

The reported precision and recall use a classification threshold of 0.20. SHAP is used to interpret model predictions by measuring feature contributions.

## 6. References

1.
