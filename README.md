# Medical Diagnosis Prediction System

This is an enhanced medical diagnosis prediction system that uses machine learning to predict heart disease, hypertension, and diabetes risk based on patient data.

## Features

1. **Risk Profile Clustering**: Groups patients into different risk profiles using unsupervised learning
2. **Multi-Disease Prediction**: Simultaneously predicts heart disease, hypertension, and diabetes risk
3. **Interactive Dashboard**: Provides comprehensive visualizations of patient data and predictions
4. **Patient History Tracking**: Saves prediction results for future reference
5. **Dual Interface**: Offers both command-line and graphical user interfaces
6. **Self-Contained**: Automatically generates sample data if no dataset is provided

## Enhanced Features Added

- **Patient History Tracking**: All predictions are saved to `patient_history.json`
- **Enhanced Visualization Dashboard**: More comprehensive charts and graphs
- **Graphical User Interface**: Easy-to-use GUI for non-technical users
- **Automatic Data Generation**: Creates sample dataset if none exists
- **Improved Error Handling**: Better error messages and recovery
- **Deployment Script**: Simplified installation and execution process

## Requirements

- Python 3.6 or higher
- Required Python packages (automatically installed by deploy.py):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

## Deployment

1. Run the deployment script:
   ```
   python deploy.py
   ```

2. This will:
   - Check your Python version
   - Install required packages
   - Create startup scripts
   - Provide instructions for running the application

## Running the Application

After deployment, you can run the application in two ways:

### Method 1: Using Startup Scripts
- **Windows**: Double-click `start_app.bat`
- **Linux/Mac**: Run `./start_app.sh`

### Method 2: Direct Execution
```
python HYbrid.py
```

## Usage

When running the application, you can choose between:
1. **Command Line Interface**: Enter patient data through text prompts
2. **Graphical User Interface**: Enter patient data through a form-based interface (if Tkinter is available)

The system will then:
1. Analyze the patient data
2. Assign them to a risk profile cluster
3. Predict their risk for heart disease, hypertension, and diabetes
4. Display comprehensive visualizations
5. Save the results to `patient_history.json`

## Output

The system provides:
- Risk profile group assignment
- Individual risk predictions for three conditions
- Comprehensive dashboard with 6 different visualizations:
  1. Patient profile context
  2. AI reliability metrics
  3. System confidence levels
  4. Key risk factors
  5. Population risk distribution
  6. Individual risk assessment

## Files

- `HYbrid.py`: Main application
- `deploy.py`: Deployment script
- `requirements.txt`: Python package dependencies
- `start_app.bat/sh`: Platform-specific startup scripts
- `heart_disease_uci.csv`: Dataset (automatically created if missing)
- `patient_history.json`: Saved prediction results

## Contributing

This is an open-source project. Feel free to contribute by:
- Reporting bugs
- Suggesting enhancements
- Submitting pull requests