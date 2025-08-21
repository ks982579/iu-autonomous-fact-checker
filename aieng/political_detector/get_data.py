import kagglehub
from datasets import load_dataset

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

print('getting more')

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("cardiffnlp/tweet_topic_multi")
# saves to ~/.cache/huggingface/datasets/...

print(ds.cache_files)