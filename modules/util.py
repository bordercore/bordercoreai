from pathlib import Path

import requests
import yaml

try:
    from trafilatura import bare_extraction
except Exception:
    # This package is not required for the API
    pass


# Load the model config file
def get_model_info():
    models_file_path = Path(__file__).resolve().parent.parent / Path("models.yaml")
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


def get_webpage_contents(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    extracted_text = bare_extraction(response.text)
    return extracted_text["raw_text"]


def strip_code_fences(text):
    """
    Some models, such as Phi-4, add an unnecessary code fence around JSON responses.
    Use this function to remove them.
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1])
    return text
