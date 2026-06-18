import numpy as np

model = np.load("Diamonds_parameters.npz")

w = model['weights']
b = model['Bias']
X_education_mean = model['X_mean']
X_education_std = model['X_std']
y_education_mean = model['y_mean']
y_education_std = model['y_std']

try:

    while True:
        carat = float(input("carat: "))
        x = float(input("X: "))
        y = float(input("Y: "))
        z = float(input("Z: "))
        cut = input("cut (Fair, Good, Very Good, Premium, Ideal): ")

        if cut == "Fair":
            cut = 1.0
        elif cut == "Good":
            cut = 2.0
        elif cut == "Very Good":
            cut = 3.0
        elif cut == "Premium":
            cut = 4.0
        elif cut == "Ideal":
            cut = 5.0
        else :
            raise ValueError("Entered invalid value!!!")
        
        color = input("Color (J,I,H,G,F,E,D): ")

        if color == "J":
            color = 1.0
        elif color == "I":
            color = 2.0
        elif color == "H":
            color = 3.0
        elif color == "G":
            color = 4.0
        elif color == "F":
            color = 5.0
        elif color == "E":
            color = 6.0
        elif color == "D":
            color = 7.0
        else:
            raise ValueError("Entered invalid value!!!")
        
        clarity = input("Clarity (I1,SI2,SI1,VS2,VS1,VVS2,VVS1,IF): ")

        if clarity == "I1":
            clarity = 1.0
        elif clarity == "SI2":
            clarity = 2.0
        elif clarity == "SI1":
            clarity = 3.0
        elif clarity == "VS2":
            clarity = 4.0
        elif clarity == "VS1":
            clarity = 5.0
        elif clarity == "VVS2":
            clarity = 6.0
        elif clarity == "VVS1":
            clarity = 7.0
        elif clarity == "IF":
            clarity = 8.0
        else:
            raise ValueError("Entered invalid value!!!")
        
        new_diamonds = np.array([[carat,carat ** 2,x,y,z,cut,color,clarity]])

        new_diamonds_norm = (new_diamonds - X_education_mean) / X_education_std

        Price_guess = np.dot(new_diamonds_norm,w) + b

        Price_guess_real = (Price_guess * y_education_std) + y_education_mean

        print(Price_guess_real)

        break
except ValueError as e:
    print(f"System error : {e}")


