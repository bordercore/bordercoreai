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
