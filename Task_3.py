from Modules import load_data, train_all_digits, classification_of_digit
import numpy as np

train_digits, train_labels, test_digits, test_labels = load_data()
bases = train_all_digits(train_digits, train_labels, n_samples=400, k_max=15)
k_values = range(5, 16)
num_k = len(k_values)
overall_accuracies = np.zeros(num_k)
digit_accuracies = np.zeros((10, num_k))  

for idx, k in enumerate(k_values):
    predictions = classification_of_digit(bases, test_digits, k)
    overall_acc = np.mean(predictions == test_labels)
    overall_accuracies[idx] = overall_acc
    print(f"k = {k:2d}   Overall accuracy: {overall_acc*100:.2f}%")
    for d in range(10):
        mask = (test_labels == d)                     
        digit_acc = np.mean(predictions[mask] == d)   
        digit_accuracies[d, idx] = digit_acc
        print(f"digit {d}: {digit_acc*100:.2f}%")