# Custom Machine Learning Pipeline: Diamond Price Prediction from Scratch

A pure mathematical implementation of a **Multiple Linear Regression** model with **Vectorized Gradient Descent Optimization** built completely from scratch using only `NumPy` and `Pandas`. This project avoids high-level machine learning frameworks (like Scikit-Learn) to demonstrate deep foundational knowledge of data preprocessing, feature engineering, mathematical optimization, and inference serialization.

---

## 📊 Model Performance Metrics

The model was trained on a 54,000+ row dataset using an 80% training and 20% validation split. Because of the mathematical foundations applied correctly, the model achieved optimal convergence:

* **$R^2$ Score (Coefficient of Determination):** `0.9067` (The model explains **90.67%** of the variance in diamond prices).
* **Mean Absolute Error (MAE):** `$820.21`
* **Root Mean Squared Error (RMSE):** `$1230.65`

---

## 🛠️ Core Engineering & Optimization Features

### 1. Advanced Mathematical Feature Engineering
Real-world diamond prices scale non-linearly with weight (carat). To address this exponential relationship within a linear architecture, a non-linear feature transformation was engineered manually:
$$X_{\text{carat\_squared}} = X_{\text{carat}}^2$$
This inclusion allowed the linear boundary to bend around quadratic trends, significantly dropping the error metrics.

### 2. Standardized Feature Scaling (Z-Score Normalization)
To prevent features with large ranges (e.g., price, dimensions) from dominating the gradient update updates over categorical features (e.g., cut quality, clarity index), strict Z-Score normalization was enforced:
$$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$
*Crucial Design Pattern:* To prevent data leakage, the mean ($\mu$) and standard deviation ($\sigma$) parameters were computed strictly from the training partition and then applied to scale the test/inference data.

### 3. Vectorized Gradient Descent Optimizer
The weights and bias parameters were trained iteratively over 25,000 steps using a vectorized batch gradient descent approach:

* **Prediction Vector:** $$\hat{y} = X \cdot w + b$$
* **Cost Function ($J$):** Mean Squared Error (MSE) minimization.
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$
* **Partial Gradients:**
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T \cdot (\hat{y} - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

---

## 📂 Project Structure & Script Architecture

* **`Diamonds_price_prediction_screen.py`**: The core data pipeline and matrix training routine. Maps qualitative factors (Cut, Color, Clarity) to linear ordinal scales, shuffles data matrix entries randomly to destroy structural bias, applies normalization, minimizes global cost via analytical gradients, and evaluates final performance metrics.
* **`Education_diamonds_price_prediction_ai.py`**: The production runtime module. It deserializes model configurations asynchronously, captures user console inputs, maps ordinal values safely, runs identical vector transformations, and executes swift inference computations.
* **`Diamonds_parameters.npz`**: Compressed model binary storing trained parameter tensors (`weights`, `Bias`, scaling matrices) for standalone zero-dependency deployment.

---

## ⚙️ Execution & Standalone Deployment

### 1. Training Phase:
Execute this script to train the model, compute normalization parameters, and serialize the optimal weights.vcxv
```bash
# Required Dependencies: numpy, pandas
python Diamonds_price_prediction_screen.py
Output: Logs structural convergence, mathematical error metrics (R², MAE, RMSE), and serializes Diamonds_parameters.npz to disk.
```

##^# 2. Inference Mode (Real-Time Interactive Prediction):
Execute this standalone module to run predictions against the pre-trained model binary.
```bash
# Required Dependencies: numpy
python Education_diamonds_price_prediction_ai.py
Output: Asks user for console inputs (carat, cut, color, etc.) and provides instant price prediction.
```
