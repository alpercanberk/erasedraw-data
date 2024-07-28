#!/bin/bash

# Create the dataset-v1 directory
mkdir -p dataset-v1
cd dataset-v1

# Base URL
BASE_URL="https://erasedraw.s3.us-east-2.amazonaws.com/dataset-v1"

# List of files to download
FILES=(
    "erasedraw-gqa-000000-wds.tar"
    "erasedraw-gqa-000001-wds.tar"
    "erasedraw-gqa-000002-wds.tar"
    "erasedraw-gqa-000003-wds.tar"
    "erasedraw-gqa-000004-wds.tar"
    "erasedraw-gqa-000005-wds.tar"
    "erasedraw-gqa-000006-wds.tar"
    "erasedraw-gqa-000007-wds.tar"
    "erasedraw-gqa-000008-wds.tar"
    "erasedraw-gqa-000009-wds.tar"
    "erasedraw-gqa-000010-wds.tar"
    "erasedraw-gqa-000011-wds.tar"
    "erasedraw-openimages-gpt4v-000000-wds.tar"
    "erasedraw-openimages-gpt4v-000001-wds.tar"
    "erasedraw-openimages-gpt4v-000002-wds.tar"
    "erasedraw-openimages-gpt4v-000003-wds.tar"
    "erasedraw-openimages-gpt4v-000004-wds.tar"
    "erasedraw-openimages-gpt4v-000005-wds.tar"
    "erasedraw-openimages-gpt4v-000006-wds.tar"
    "erasedraw-openimages-gpt4v-000007-wds.tar"
    "erasedraw-openimages-gpt4v-000008-wds.tar"
    "erasedraw-sharegpt4v-000000-wds.tar"
    "erasedraw-sharegpt4v-000001-wds.tar"
    "erasedraw-sharegpt4v-000002-wds.tar"
)

# Download each file
for file in "${FILES[@]}"; do
    echo "Downloading $file..."
    wget "$BASE_URL/$file"
done

echo "All downloads completed."