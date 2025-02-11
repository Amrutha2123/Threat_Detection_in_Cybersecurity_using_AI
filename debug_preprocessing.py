from preprocess import preprocess_data

train_path = "../Data/Train_data.csv"
test_path = "../Data/Test_data.csv"

train_data = preprocess_data(train_path)
test_data = preprocess_data(test_path)

if train_data is not None and test_data is not None:
    print("Preprocessing completed successfully!")
else:
    print("Preprocessing failed.")

