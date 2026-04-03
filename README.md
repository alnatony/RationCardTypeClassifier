# Ration Card Type Classifier

This project predicts Kerala ration card categories using machine learning based on socio-economic and household features.

---

## Categories

- AAY (Yellow) – Poorest households  
- PHH (Pink) – Priority households  
- NPS (Blue) – Non-priority subsidy  
- NPI (Brown) – Institutional residents (nuns, priests, etc.)  
- NPNS (White) – Non-priority non-subsidy (e.g., government employees)

---

## Problem Statement

Manual classification of ration cards is time-consuming and can lead to inconsistencies.  
This project automates the classification process using a machine learning model.

---

## Methodology

1. Data collection and cleaning  
2. Feature engineering (income, employment, family details)  
3. Encoding categorical values  
4. Model training using XGBoost  
5. Prediction of ration card category  

---

## Project Structure

```
src/
├── app/        # Application logic
├── data/       # Data preprocessing scripts
├── models/     # Model training code
├── ocr/        # OCR extraction
└── tests/      # Testing scripts (NPI case)

```

---

## How to Run

1. Install dependencies  

```
pip install -r requirements.txt

```

2. Train the model  

```
python src/models/train_model.py

```

3. Run the application  

```
python src/app/app1.py

```

---

## Output

The model predicts the ration card category based on input features.

---

## Key Insight

The model correctly identifies **NPI (brown card)** as institutional residents with zero income, showing it learned real-world patterns instead of random classification.

---

## Notes

- Dataset and model files are excluded using `.gitignore`  
- The model can be regenerated using the training script  