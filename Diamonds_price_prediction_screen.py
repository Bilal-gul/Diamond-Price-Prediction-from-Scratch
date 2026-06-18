import pandas as pd
import numpy as np

#region edit data

df = pd.read_csv("diamonds.csv")

cut_quality = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
df['cut'] = df['cut'].map(cut_quality)

color_map = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
df['color'] = df['color'].map(color_map)

clarity_map = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}
df['clarity'] = df['clarity'].map(clarity_map)

df['carat_squared'] = df['carat']**2

X = df[['carat','carat_squared','x','y','z','cut','color','clarity']].values
y = df[['price']].values.reshape(-1,1)

#endregion

#region Normalization

total_lines = X.shape[0]

indexes = np.arange(total_lines)
np.random.shuffle(indexes)

X_mixed = X[indexes]
y_mixed = y[indexes]

cutting = int(total_lines * 0.8)

X_education_prop = X_mixed[:cutting]
y_education_trg = y_mixed[:cutting]

X_test_prop = X_mixed[cutting:]
y_test_trg = y_mixed[cutting:]

X_education_mean = np.mean(X_education_prop, axis=0)
X_education_std = np.std(X_education_prop, axis=0)

y_education_trg_mean = np.mean(y_education_trg,axis=0)
y_education_trg_std = np.std(y_education_trg,axis=0)

X_education_prop_norm = (X_education_prop - X_education_mean) / X_education_std
y_education_trg_norm = (y_education_trg - y_education_trg_mean) / y_education_trg_std

X_test_prop_norm = (X_test_prop - X_education_mean) / X_education_std
y_test_trg_norm = (y_test_trg - y_education_trg_mean) / y_education_trg_std

#endregion

#region update and learning

learning_rate = 0.1
m = X_education_prop.shape[0]
n = X_education_prop.shape[1]

w = np.zeros((n,1))
b = 0.0

iteration = 25000

for i in range(iteration):

    y_guess = np.dot(X_education_prop_norm,w) + b 

    
    error = y_guess - y_education_trg_norm

    dw = (1/m) * np.dot(X_education_prop_norm.T,error)
    db = (1/m) * np.sum(error)

    w = w - (learning_rate*dw)
    b = b - (learning_rate*db)

    if i%100 == 0 :
        cost_function = (1/(2*m)) * np.sum(error**2)
        print(f"Cost: {cost_function:.20f}")

print(f"Weights : {w}")

#endregion

#region dataset test
y_test_guess_norm = np.dot(X_test_prop_norm, w) + b

y_test_guess_real = (y_test_guess_norm * y_education_trg_std) + y_education_trg_mean

MAE = np.mean(np.abs(y_test_guess_real - y_test_trg))
print(f"Mean average error (MAE): ${MAE:.2f}")


RMSE = np.sqrt(np.mean((y_test_guess_real - y_test_trg)**2))
print(f"RMSE: ${RMSE:.2f}")

SS_res = np.sum((y_test_trg - y_test_guess_real)**2)
SS_tot = np.sum((y_test_trg - np.mean(y_test_trg))**2)
R2 = 1 - (SS_res / SS_tot)
print(f"R² score: {R2:.4f}")

#endregion

#region register file
np.savez("Diamonds_parameters.npz", weights = w, Bias = b, X_mean = X_education_mean, X_std = X_education_std, y_mean = y_education_trg_mean, y_std = y_education_trg_std)
print("The model was trained succesfully!!!")
#endregion
