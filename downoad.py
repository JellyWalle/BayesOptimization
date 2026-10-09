import kagglehub

# Download latest version
path = kagglehub.dataset_download("itsahmad/indoor-scenes-cvpr-2019")

print("Path to dataset files:", path)