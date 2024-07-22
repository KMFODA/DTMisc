from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import list_repo_refs, delete_tag, create_tag

config = AutoConfig.from_pretrained("distributed/gpt2-250m")
model = AutoModel.from_config(config)

tokenizer = AutoTokenizer.from_pretrained("distributed/gpt2-250m")

model.push_to_hub("kmfoda/gpt2-250m")
tokenizer.push_to_hub("kmfoda/gpt2-250m")
create_tag(
    "kmfoda/gpt2-250m",
    repo_type="model",
    tag="0",
    tag_message=f"Epcoh 0",
)
