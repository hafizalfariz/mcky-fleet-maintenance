# MCKY – Maintenance Check and Safety for Fleet

> Predictive maintenance system to monitor fleet health, detect potential issues, and provide actionable insights for logistics companies.

---

## Repository Outline

1. **README.md** – Project overview & documentation.  
2. **notebooks/** – Jupyter Notebooks for the ML pipeline (EDA, modeling, inference):
   - `01_ModelCNN_Tire_Textures_EDA.ipynb` – EDA for tire textures (CV)
   - `02_ModelCNN_Tire_Textures_Model.ipynb` – Model development (CV)
   - `03_ModelCNN_Tire_Textures_Inf.ipynb` – Inference (CV)
   - `final_project_RMT45_01.ipynb` – Main notebook (EDA, regression, NLP, CV)
   - `final_project_RMT45_01_inference.ipynb` – Inference pipeline
   - `Model_Classification.ipynb` – Classification model (CV)
   - `Model_Classification_Inference.ipynb` – Inference (CV)
3. **back-end/** – Backend API (Flask):
   - `app.py` – API for model inference
4. **front-end/** – Frontend web app (HTML/JS/CSS):
   - `index.html` – Main page
   - `assets/` – Static assets (images, CSS, JS, fonts)
5. **images/** – Diagrams, flowcharts, dashboard screenshots (e.g., `flow_aplication.png`, `1.png`, `2.png`)
6. **data/** – Raw and processed fleet datasets (engine sensors, operational data, user feedback).  
7. **models/** – Saved machine learning models:  
   - `regression_model.pkl` – Predictive maintenance (engine/operational data).  
   - `nlp_model.pkl` – Complaint text classification.  
   - `cv_model.pkl` – Image-based vehicle condition classification.  
8. **deployment/**  
   - `preprocessing_pipeline.pkl` – Data preprocessing pipeline.  
   - `streamlit_app.py` – Frontend deployment (Streamlit/Hugging Face).  
   - `eda_app.py` – EDA & visualization app.  

---

## Problem Background

Fleet management in logistics companies faces high operational costs due to unexpected vehicle breakdowns and inefficient maintenance scheduling.  
By combining engine sensor data, operational load information, and user feedback, predictive models can anticipate potential failures, reduce downtime, and improve overall fleet safety.

---

## Project Output

- **Predictive Maintenance Model (Regression):** Classifies vehicle condition → Good / Needs Investigation / Needs Maintenance.  
- **NLP Model:** Processes user complaints (text) to identify potential mechanical issues.  
- **Computer Vision Model:** Classifies image-based input (vehicle/battery/engine condition) from users.  
- **Dashboard:** Provides alerts and recommendations (go to mechanic or update status).  
- **Deployed Web App:** Interactive prediction and EDA hosted on [Hugging Face](https://huggingface.co/spaces/HelasOn7/fe-finpro).  

---

## Data

- **Source**: Logistics Vehicle Maintenance Dataset (Kaggle) & simulated user complaint/image data.  
- **Features**:  
  - **Engine/Sensor Data (Div. Armada):** Engine temperature, tire pressure, usage hours, vibration levels, battery status, vehicle info.  
  - **Operational Data (Div. Operasional):** Actual load, route info, historical maintenance, last maintenance date.  
  - **User Feedback:** Complaint texts, uploaded images.  
- **Target:** Maintenance status → Good / Investigate / Maintenance required.  

---

## Method

- **Regression Models** – Predict vehicle condition based on numerical and categorical features.  
- **NLP Models** – Text classification for complaint data.  
- **Computer Vision Models** – Image classification for physical damage/condition reports.  
- **Dashboard Integration** – Centralized platform showing alerts and recommended actions.  
- **Evaluation Metrics** – Accuracy, Precision, Recall, F1-score for classification tasks; RMSE/R² for regression tasks.  

---


## System Flow
![System Flow](images/flowchart.png)

**System Flow Overview:**

1. **Data Input:**
   - *Div. Armada* uploads engine sensor data (temperature, tire pressure, usage hours, vibration, battery, etc).
   - *Div. Operasional* uploads operational data (load, route, maintenance history).
   - *User* submits feedback (complaint text, images).

2. **Preprocessing:**
   - Data is cleaned and transformed using the preprocessing pipeline.

3. **Prediction:**
   - **Regression Model:** Predicts vehicle condition (Good / Investigate / Maintenance).
   - If result is **Investigate**:
     - **NLP Model:** Classifies complaint text for potential issues.
     - **CV Model:** Classifies uploaded images for physical condition.

4. **Dashboard & Alerts:**
   - Results are displayed on the dashboard (frontend web app / Streamlit).
   - Alerts and recommendations are generated (e.g., send to workshop, update status).

5. **Action:**
   - If "Maintenance" → vehicle is sent to workshop.
   - If "Good" → status is updated, vehicle continues operation.

**See `images/flow_aplication.png` for the detailed flowchart.**

---

## Stacks

- **Python 3.x**  
- Jupyter Notebook  
- Pandas, NumPy  
- Scikit-learn, XGBoost, LightGBM  
- TensorFlow / PyTorch (for NLP & CV)  
- Matplotlib, Seaborn, Plotly  
- Streamlit (deployment)  
- Hugging Face Spaces (app hosting)  

---

## Reference

- Kaggle: Logistics Vehicle Maintenance Dataset  
- Hacktiv8 Machine Learning & Deployment Materials  
- Automotive & logistics maintenance case studies  

---

## Additional References

- [Hugging Face App – MCKY](https://huggingface.co/spaces/HelasOn7/fe-finpro)  

 

![Dashboard Screenshot](images/1.png)
![Homepage](images/2.png)
---

## Contact

For questions or collaboration:  
**Hafiz Alfariz** – [LinkedIn](https://www.linkedin.com/in/hafizalfariz/) | [GitHub](https://github.com/hafizalfariz)

**Fhad Saleh** - [LinkedIn](https://www.linkedin.com/in/fhad-saleh-5b4761168/) | [GitHub](https://github.com/helason7)

