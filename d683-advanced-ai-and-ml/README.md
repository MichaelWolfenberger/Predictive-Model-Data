##Advanced AI and ML - CO2 Emission Predictor

## Software & Hardware Requirements
**Hardware Requirements:**
- Operating System: Windows 10/11, macOS, or Linux
- RAM: Minimum 8GB (16GB recommended)
- Processor: Multi-core CPU (e.g., Intel i5 or AMD equivalent)
- Storage: At least 500MB of free space

**Software Requirements:**
- Python 3.12 (or compatible 3.x version)
- IDE: IntelliJ IDEA, PyCharm, or VS Code
- Required Python Libraries (see `requirements.txt`):
    - pandas
    - scikit-learn
    - xgboost
    - shap
    - matplotlib

## Instructions to Run the Application
1. Ensure Python is installed on your system.
2. Navigate to the root directory of this project in your terminal.
3. Activate your virtual environment (if using one).
4. Install the required dependencies by running: `pip install -r requirements.txt`
5. Navigate to the scripts folder: `cd scripts`
6. Execute the model script: `python model.py`
7. The terminal will output the GridSearchCV optimal parameters, cross-validation scores, and the final classification metrics (Accuracy, Precision, Recall, F1 Score). A SHAP summary plot visualization will also open automatically.
