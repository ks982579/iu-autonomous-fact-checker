import kagglehub

REQUIRED = False

if REQUIRED:
    # Dowload latest version
    path = kagglehub.dataset_download(
        "kaushiksuresh147/political-tweets"
    )
    print("Path to dataset files: ", path)

    # https://www.kaggle.com/datasets/crowdflower/political-social-media-posts?select=political_social_media.csv
    path = kagglehub.dataset_download("crowdflower/political-social-media-posts")
    print("Path to dataset files:", path)
else:
    print("Data should be in project already")
