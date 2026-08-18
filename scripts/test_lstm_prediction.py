import numpy as np
from src.models.lstm_forecast import predict_lstm

# seq shape (20, 3)
seq = np.random.rand(20, 3).astype(np.float32)
y_next = predict_lstm(seq)
print(y_next)