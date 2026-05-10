import numpy as np
import matplotlib.pyplot as plt

#load data from map HandwrittenDigits
def load_data():
    train_digits = np.load('HandwrittenDigits/TrainDigits.npy')
    train_labels = np.load('HandwrittenDigits/TrainLabels.npy').ravel() # get 1d
    test_digits = np.load('HandwrittenDigits/TestDigits.npy')
    test_labels = np.load('HandwrittenDigits/TestLabels.npy').ravel()
    return train_digits, train_labels, test_digits, test_labels


# compute svd and get out U,S,V_T
def compute_svd(train_digits, train_labels, digit, n_samples =400):
    a = train_labels == digit
    digit_image = train_digits[:,a][:,:n_samples]
    U,S, V_T = np.linalg.svd(digit_image, full_matrices=False)
    return U, S, V_T


def plot_singular_images(U,digit):
    fig, axs = plt.subplots(1,3, figsize = (10,4)) # 3 images in 1 big image
    for i, ax in enumerate(axs):
        u = U[:,i]
        img = np.reshape(u,(28,28)).T # transform the vectror to get a nice picture
        ax.imshow(img,cmap = 'gray')
        ax.set_title(f'$u_{i+1}$ ', fontsize=16)
        ax.axis('off')
    fig.suptitle(f'Singular images of digit {digit}', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_singular_values(S,digit):
    plt.figure()
    plt.plot(S[:50], 'o') # plot the first 50 singular values, the rest follow the decreasing trend
    plt.title(f'Plot over singular values for digit {digit}', fontsize=16)
    plt.xlabel('Index', fontsize=16)
    plt.ylabel('Singular value', fontsize=16)
    plt.tick_params(axis='both', labelsize=16) 
    plt.grid(True)
    plt.show()

def train_all_digits(train_digits, train_labels, n_samples=400,k_max=15): 
    bases = {} # empty dict to store bases to corresponding digit
    for d in range(10):
        U,_,_ = compute_svd(train_digits, train_labels, digit=d, n_samples=n_samples)
        bases[d] = U[:, :k_max] # only save k_max number of base-vectors
    return bases

def classification_of_digit(bases,test_digits,k):
    N = 40000 # all pictures 
    r_matrix = np.zeros((10,N))
    D = test_digits 
    for d in range(10):
        B_d = bases[d]
        U_k = B_d[:,:k]
        P = U_k @ (U_k.T @ D) # D matrix with all test digits
        R = D -P        # residual matrix
        residual_norm = np.linalg.norm(R, axis=0) 
        r_matrix[d,:] = residual_norm
    return np.argmin(r_matrix, axis=0) # only returns the smallest - ie guesses for what digit it is





    
