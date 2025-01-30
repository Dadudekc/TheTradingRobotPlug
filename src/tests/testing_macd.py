from sklearn.metrics import mean_squared_error

y_true = [1, 2, 3]
y_pred = [1.1, 1.9, 3.2]

# Test with 'squared=False'
rmse = mean_squared_error(y_true, y_pred, squared=False)
print(f"RMSE: {rmse}")
