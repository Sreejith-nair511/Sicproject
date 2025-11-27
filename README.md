# Medical Diagnosis Prediction System

This is an enhanced medical diagnosis prediction system that uses machine learning to predict heart disease, hypertension, and diabetes risk based on patient data. The system combines powerful data analysis with an intuitive user interface to help medical professionals quickly assess patient risk profiles.

##  Key Features

###  Risk Profile Clustering
- Groups patients into different risk profiles using unsupervised learning (K-Means clustering)
- Helps identify patterns and similarities between patients
- Visualizes where a patient fits within the broader population

###  Multi-Disease Prediction
- Simultaneously predicts three major health conditions:
  - Heart Disease
  - Hypertension (High Blood Pressure)
  - Diabetes Risk
- Uses advanced Random Forest algorithms for accurate predictions

### Interactive Dashboard
- Provides comprehensive visualizations of patient data and predictions
- Shows 6 different charts for complete insight into patient health
- Easy-to-understand graphs and metrics for medical professionals

###  Patient History Tracking
- Automatically saves all prediction results for future reference
- Maintains a detailed history in `patient_history.json`
- Helps track patient progress over time

###  Dual Interface Options
- **Command Line Interface**: For technical users who prefer keyboard input
- **Graphical User Interface**: Easy-to-use form-based interface for all users
- Choose your preferred method when starting the application

###  Self-Contained System
- Automatically generates sample data if no dataset is provided
- Works out-of-the-box with minimal setup
- No external dependencies required to get started

##  Enhanced Features

- **Patient History Tracking**: All predictions are automatically saved to `patient_history.json`
- **Enhanced Visualization Dashboard**: Comprehensive charts and graphs for better insights
- **Graphical User Interface**: Easy-to-use GUI for non-technical users
- **Automatic Data Generation**: Creates sample dataset if none exists
- **Improved Error Handling**: Better error messages and recovery mechanisms
- **Deployment Script**: Simplified installation and execution process

##  Requirements

- Python 3.6 or higher
- Required Python packages (automatically installed by deploy.py):
  - pandas (data manipulation)
  - numpy (numerical computing)
  - matplotlib (plotting and visualization)
  - seaborn (statistical data visualization)
  - scikit-learn (machine learning)

##  Installation & Deployment

### Quick Setup
1. Run the deployment script:
   ```
   python deploy.py
   ```

2. The deployment process will automatically:
   - Check your Python version
   - Install all required packages
   - Create convenient startup scripts
   - Provide clear instructions for running the application

### Manual Installation
If you prefer to install manually:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

##  Running the Application

After deployment, you can run the application in multiple ways:

### Method 1: Using Startup Scripts (Easiest)
- **Windows**: Double-click `start_app.bat`
- **Linux/Mac**: Run `./start_app.sh`

### Method 2: Direct Execution
```
python HYbrid.py
```

##  How to Use

When you start the application, you'll be prompted to choose between two interfaces:

### Command Line Interface (Option 1)
1. Enter patient information when prompted:
   - Age
   - Sex (M/F)
   - Blood Pressure
   - Cholesterol Level
   - Maximum Heart Rate
   - Fasting Blood Sugar (> 120 mg/dl)

2. Receive immediate analysis and risk predictions

### Graphical User Interface (Option 2)
1. Fill in the patient information in the easy-to-use form
2. Click "Predict Risk" to analyze the data
3. View results in a popup window

##  Output & Results

The system provides comprehensive analysis including:

### Risk Profile Assignment
- Assigns the patient to one of three risk profile clusters
- Shows where the patient fits within the broader population

### Individual Risk Predictions
- Heart Disease Risk: Positive (Risk Detected) or Negative (Healthy)
- Hypertension Risk: Positive (Risk Detected) or Negative (Healthy)
- Diabetes Risk: Positive (Risk Detected) or Negative (Healthy)

### Visualization Dashboard
The system generates 6 detailed visualizations:
1. **Patient Profile Context**: Shows where the patient fits among all profiles
2. **AI Reliability Metrics**: Confusion matrix showing prediction accuracy
3. **System Confidence Levels**: Bar chart showing confidence for each prediction
4. **Key Risk Factors**: Horizontal bar chart showing most important factors
5. **Population Risk Distribution**: Pie chart showing overall risk distribution
6. **Individual Risk Assessment**: Visual representation of patient's specific risks

### Data Storage
- All predictions are automatically saved to `patient_history.json`
- Each entry includes timestamp, patient data, and prediction results

##  Project Files

- `HYbrid.py`: Main application with both CLI and GUI interfaces
- `deploy.py`: Automated deployment and setup script
- `requirements.txt`: Python package dependencies
- `start_app.bat/sh`: Platform-specific startup scripts
- `heart_disease_uci.csv`: Medical dataset (automatically created if missing)
- `patient_history.json`: Saved prediction results and patient history
- `setup.bat`: Windows setup script

##  Contributing

This is an open-source project designed to help medical professionals and researchers. Contributions are welcome!

You can contribute by:
- Reporting bugs or issues
- Suggesting new features or enhancements
- Submitting pull requests with improvements
- Sharing the project with others in the medical community

##  Support

If you encounter any issues or have questions about the system:
1. Check that all requirements are installed
2. Ensure you're using Python 3.6 or higher
3. Run the deploy.py script to reinstall dependencies
4. Contact the development team through GitHub issues

## 📄 License

This project is open-source and available for use in medical research and practice. 
Please cite appropriately if used in academic or clinical settings.
