if __name__ == "__main__":
    import os
    import mnist

    cache_directory = os.path.join(os.getcwd(), ".cache")

    dataset = mnist.Dataset(cache_directory)
    dataset.download()

    train_images = dataset.load_train_images()
    train_labels = dataset.load_train_labels()
    test_images = dataset.load_test_images()
    test_labels = dataset.load_test_labels()

    print("Device", train_images.device().platform)
