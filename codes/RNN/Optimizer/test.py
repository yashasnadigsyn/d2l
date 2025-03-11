import pandas as pd
import numpy as np

data = pd.read_csv("../tesla-stock-price.csv")
tesla_stock = list(data["open"])[:400]
X_t = np.arange(-20, 20, 0.1)
X_t = X_t.reshape(len(X_t), 1)
Y_t = np.array(tesla_stock).reshape(len(X_t), 1)

print(tesla_stock)

