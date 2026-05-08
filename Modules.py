import numpy as np
import matplotlib.pyplot as plt

# 1. Ladda data
def load_data():
    train_digits = np.load('HandwrittenDigits/TrainDigits.npy')
    train_labels = np.load('HandwrittenDigits/TrainLabels.npy').ravel() # get 1d
    test_digits = np.load('HandwrittenDigits/TestDigits.npy')
    test_labels = np.load('HandwrittenDigits/TestLabels.npy').ravel()
    return train_digits, train_labels, test_digits, test_labels

def compute_svd(train_digits, train_labels, digit, n_samples =400):
    a = train_labels == digit
    digit_image = train_digits[:,a][:,:n_samples]
    U,S, V_T = np.linalg.svd(digit_image, full_matrices=False)
    return U, S, V_T

def plot_singular_images(U,digit):
    fig, axs = plt.subplots(1,3, figsize = (10,4))
    for i, ax in enumerate(axs):
        u = U[:,i]
        img = np.reshape(u,(28,28)).T
        ax.imshow(img,cmap = 'gray')
        ax.set_title(f'$u_{i+1}$ ')
        ax.axis('off')
    fig.suptitle(f'Singular images of digit {digit}')
    plt.tight_layout()
    plt.show()

def plot_singular_values(S,digit):
    plt.figure()
    plt.plot(S[:50], 'o')
    plt.title(f'Plot over singular values {digit}')
    plt.xlabel('Index')
    plt.ylabel('Singular value')
    plt.grid(True)
    plt.show()

def train_all_digits(train_digits, train_labels, n_samples=400,k_max=15): 
    bases = {}
    for d in range(10):
        U,_,_ = compute_svd(train_digits, train_labels, digit=d, n_samples=n_samples)
        bases[d] = U[:, :k_max]
    return bases

def classification_of_digit(bases,test_digits,k):
    N = 40000
    r_matrix = np.zeros((10,N))
    D = test_digits
    for d in range(10):
        B_d = bases[d]
        U_k = B_d[:,:k] # k - how many digits we test
        P = U_k @ (U_k.T @ D) # D matrix with all test digits
        R = D -P        # residual matrix
        residual_norm = np.linalg.norm(R, axis=0) 
        r_matrix[d,:] = residual_norm
    return np.argmin(r_matrix, axis=0)




 










    ''' 
def train_data(train_digits, train_labels, digit, n_samples =400, k_max=100):
    a = train_labels == digit
    digit_image = train_digits[:,a][:,:n_samples]
    U,S, V_T = np.linalg.svd(digit_image, full_matrices=False)
    basis = U[:,:k_max]
    # ploten bör läggas i en egen funktion
    plt.figure()
    plt.title('40 first Singular Values')
    plt.xlabel('Index')
    plt.ylabel(' Singular Values')
    
    plt.plot(S[:40], marker = 'o')
    plt.legend()
    plt.grid()
    plt.show()
    

    return basis, S, V_T
'''







    
