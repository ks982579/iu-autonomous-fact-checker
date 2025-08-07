import kagglehub

REQUIRED = False

if REQUIRED:
    # Dowload latest version
    path = kagglehub.dataset_download(
        "kaushiksuresh147/political-tweets"
    )

    print("Path to dataset files: ", path)
else:
    print("Data should be in project already")
