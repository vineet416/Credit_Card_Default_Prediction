# Credit Card Default Prediction - ([App_Link](https://credit-card--default-prediction.streamlit.app/))

A comprehensive machine learning project to predict credit card default payments using various classification algorithms. This project implements an end-to-end pipeline from data ingestion to model deployment using best practices in MLOps.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Web Application](#web-application)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🎯 Project Overview

In an increasingly dynamic financial landscape, XYZ Financial Services faces the critical challenge of accurately predicting credit risk. This project develops a predictive model that estimates the probability of credit default based on credit card owners' characteristics such as age, gender, education, marital status, credit limit, and payment history.

### Business Goal
Enable XYZ Financial Services to:
- Identify high-risk credit clients
- Tailor risk mitigation strategies
- Adjust credit limits appropriately
- Offer targeted financial counseling
- Reduce default rates and improve portfolio health

## 📊 Dataset Information

- **Source**: UCI Machine Learning Repository ([Data Source Link](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset))
- **Dataset**: UCI_Credit_Card.csv
- **Size**: 30,000 instances with 25 features
- **Target Variable**: `default.payment.next.month` (Binary: 1 = default, 0 = no default)
- **Time Period**: April 2005 to September 2005 (Taiwan)

### Key Features
- **Demographic**: Age, Gender, Education, Marital Status
- **Credit Information**: Credit Limit Balance
- **Repayment Status**: Past 6 months payment status (PAY_0 to PAY_6)
- **Bill Statements**: Past 6 months bill amounts (BILL_AMT1 to BILL_AMT6)
- **Payment Amounts**: Past 6 months payment amounts (PAY_AMT1 to PAY_AMT6)

## ✨ Features

- **Data Pipeline**: Automated data ingestion, transformation, and preprocessing
- **Model Training**: Multiple algorithm comparison and hyperparameter tuning
- **Model Evaluation**: Comprehensive performance metrics and visualization
- **Web Application**: Interactive Streamlit app for real-time predictions
- **Logging**: Comprehensive logging system for debugging and monitoring
- **Exception Handling**: Robust error handling throughout the pipeline
- **Modular Architecture**: Clean, maintainable, and scalable codebase

## 📁 Project Structure

```
Credit Card Default Prediction/
├── artifacts/                          # Artifacts folder stores all the outputs of ML pipeline
│   ├── credit_data.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── test.csv
│   └── train.csv
├── config/                            # Configuration files
│   └── model.yaml
├── logs/                              # Application logs
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── 1-exploratory_data_analysis-EDA.ipynb
│   ├── 2-data_preprocessing.ipynb
│   ├── 3-model_training_and_evaluation.ipynb
│   ├── csv_outputs/                   # Model performance results
│   ├── datasets/                      # Raw dataset
│   ├── feature_importance_outputs/    # Feature importance plots
│   ├── test_performance_outputs/      # Test performance visualizations
│   └── validation_performance_outputs/ # Validation performance visualizations
├── src/                               # Source code
│   ├── components/                    # Core components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/                      # Training and prediction pipelines
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── constant/                      # Constants
│   ├── utils/                         # Utility functions
│   ├── exception.py                   # Custom exception handling
│   └── logger.py                      # Logging configuration
├── streamlit_app.py                   # Web application
├── upload_data.py                     # Upload data into MongoDB
├── requirements.txt                   # Dependencies
├── setup.py                          # Package setup
└── README.md                         # Project documentation
```

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/vineet416/Credit_Card_Default_Prediction.git
cd Credit_Card_Default_Prediction
```

2. **Create a virtual environment**:
```bash
conda create -p venv python==3.12 -y
conda activate venv/
```

3. **Install dependencies and package**:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

1. **Run the training pipeline**:
```python
from src.pipeline.train_pipeline import TrainPipeline

# Initialize and run training pipeline
pipeline = TrainPipeline()
pipeline.run_pipeline()
```

2. **Or run via command line**:
```bash
python src/pipeline/train_pipeline.py
```

### Web Application

Launch the Streamlit web application:
```bash
streamlit run streamlit_app.py
```

The web app provides an intuitive interface for:
- Input credit card holder information
- Real-time default probability prediction
- Interactive feature input with explanations

## 📈 Model Performance

The project evaluates multiple machine learning algorithms:

| Model | ROC AUC | Precision | Recall | F1 Score | Accuracy |
|-------|---------|-----------|--------|----------|----------|
| **Random Forest** | 0.777 | 0.57 | 0.483 | 0.523 | 0.805 |
| **Gradient Boosting** | 0.759 | 0.585 | 0.41 | 0.483 | 0.805 |
| **XGBoost** | 0.758 | 0.504 | 0.515 | 0.509 | 0.781 |
| **K-Nearest Neighbors** | 0.693 | 0.375 | 0.521 | 0.436 | 0.702 |

**Best Model**: Random Forest with the highest ROC AUC score of 0.777 and F1 Score of 0.523.

### Key Insights:
- Random Forest provides the best balance of precision and recall
- All models achieve good Areas under the ROC curve around 0.7
- Feature importance analysis reveals payment history as the most predictive factor

## 🌐 Web Application - ([App_Link](https://credit-card--default-prediction.streamlit.app/))

The Streamlit web application includes:

- **User-friendly Interface**: Intuitive input fields for all features
- **Real-time Predictions**: Instant default probability calculation
- **Feature Explanations**: Detailed descriptions of input parameters
- **Interactive Visualizations**: Dynamic charts and plots
- **Responsive Design**: Works on desktop and mobile devices

### Application Features:
- Basic information input (age, gender, education, marital status)
- Credit limit and payment history tracking
- Bill amounts and payment amounts for past 6 months
- Instant prediction results with probability scores
- Visualizations of top 5 features influencing the prediction

## 🛠️ Technologies Used

- **Programming Language**: Python 3.12+
- **Machine Learning**: scikit-learn, XGBoost, imbalanced-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Streamlit
- **Configuration**: PyYAML
- **Database**: pymongo (MongoDB)
- **Development**: Jupyter Notebooks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

**Vineet Patel**
- Email: vineetpatel468@gmail.com
- GitHub: [@vineet416](https://github.com/vineet416)
- LinkedIn: [@vineet416](https://www.linkedin.com/in/vineet416/)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- The open-source community for the amazing tools and libraries
- Streamlit for the web application framework and ease of deployment

---

⭐ If you found this project helpful, please give it a star!
