from pathlib import Path

import yaml
from transformers import AutoTokenizer


def get_tokenizer(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model.resize_token_embeddings(len(tokenizer))

    # Add tokens <|im_start|> and <|im_end|>; the latter is special eos token
    # self.tokenizer.add_tokens(["<|im_start|>"])
    # self.tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))

    return tokenizer


# Load the model config file
def get_model_info():

    models_file_path = Path(__file__).resolve().parent / Path("models.yaml")
    with open(models_file_path, "r") as file:
        model_info = yaml.safe_load(file)

    return model_info


# Sort models based on where they are listed in the models.yaml file
def sort_models(original_list, sort_order):
    # Create a dictionary to hold the sort order with their indices
    sort_order_dict = {value: index for index, value in enumerate(sort_order)}

    # Split the original list into items to sort and items to keep at the end
    to_sort = [item for item in original_list if item["name"] in sort_order_dict]
    to_keep = [item for item in original_list if item["name"] not in sort_order_dict]

    # Sort the items to sort based on the sort order
    sorted_items = sorted(to_sort, key=lambda x: sort_order_dict[x["name"]])

    # Combine the sorted items and the items to keep at the end
    return sorted_items + to_keep
