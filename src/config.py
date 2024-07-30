from exllamav2 import ExLlamaV2Config
from huggingface_hub import snapshot_download
import os

env_repo_id = os.getenv("REPO_ID")
assert env_repo_id is not None, "REPO_ID must be set to the HuggingFace repo id of your exl2 model."
assert not env_repo_id.startswith("http"), "REPO_ID should look like `TheBloke/Llama-2-7B-Chat-GGUF`, not a URL."

env_revision = os.getenv("REVISION")
assert env_revision is not None, "REVISION should be the specific HuggingFace revision of your exl2 model (eg, 8.0bpw) "

def get_config(repo_id=env_repo_id, revision=env_revision):
	dir = snapshot_download(repo_id=repo_id, revision=revision)
	config = ExLlamaV2Config(model_dir=dir)
	return config