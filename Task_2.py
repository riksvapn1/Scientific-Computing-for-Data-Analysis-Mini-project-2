from Modules import load_data, compute_svd, plot_singular_images, plot_singular_values

#----Task 2-

train_imgs, train_lbls, _, _ = load_data()

for d in [3, 8]:
    U, S, _ = compute_svd(train_imgs, train_lbls, digit=d, n_samples=400)
    plot_singular_images(U,d)
    plot_singular_values(S,d)

