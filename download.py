from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepwaterhorizon/minecraft-skin-model",
    local_dir="models/minecraft-skin-model",
    # only grab the final pipeline pieces:
    allow_patterns=[
        "model_index.json",
        "unet/*",
        "vae/*",
        "text_encoder/*",
        "tokenizer/*",
        "scheduler/*",
        "feature_extractor/*",
    ],
)