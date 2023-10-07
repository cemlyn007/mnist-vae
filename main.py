if __name__ == "__main__":
    import os
    import mnist
    import jax

    device = jax.devices()[0]

    cache_directory = os.path.join(os.getcwd(), ".cache")

    dataset = mnist.Dataset(cache_directory)
    dataset.download()

    train_images = dataset.load_train_images(device)
    train_labels = dataset.load_train_labels(device)
    test_images = dataset.load_test_images(device)
    test_labels = dataset.load_test_labels(device)

    print("Device", train_images.device().platform)
