from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import list_repo_refs, delete_tag, create_tag

config = AutoConfig.from_pretrained("distributed/gpt2-250m")
model = AutoModel.from_config(config)

tokenizer = AutoTokenizer.from_pretrained("distributed/gpt2-250m")

model.push_to_hub("distributed/gpt2-250m-convergence-test")
tokenizer.push_to_hub("distributed/gpt2-250m-convergence-test")
create_tag(
    "distributed/gpt2-250m-convergence-test",
    repo_type="model",
    tag="0",
    tag_message=f"Epcoh 0",
)
