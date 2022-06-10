import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, RobertaTokenizer, RobertaTokenizerFast


# transforms
class Tokenization(torch.nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, inputs):
        """Tokenize the input text.

        Args:
            inputs (List[str]): A pair of raw texts to be tokenized.
        """
        return [self.tokenizer.tokenize(text) for text in inputs]


class Encoding(torch.nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        super().__init__()
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_length = max_length

    def forward(self, inputs):
        """Encoding the inputs.

        Args:
            inputs (List[List[str]]]): A pair of token sequences.
        """
        if isinstance(self.tokenizer, RobertaTokenizerFast) or isinstance(self.tokenizer, RobertaTokenizer):
            inputs = [self.tokenizer.convert_tokens_to_string(inputs[i]) for i in range(len(inputs))]

            return self.tokenizer(inputs[0], inputs[1],
                                  add_special_tokens=True,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True)

        return self.tokenizer(inputs[0], inputs[1],
                              is_split_into_words=True,
                              add_special_tokens=True,
                              max_length=self.max_length,
                              padding='max_length',
                              truncation=True)


# randomly delete `n_removal` elements from tokens
class RandomDeletion(torch.nn.Module):
    """The custom module for random deletion implementation.

    Args:
        tokenizer (PreTrainedTokenizerBase): Any of huggingface tokenizers that inherits PreTrainedTokenizerBase.
        rate (float): The removal rate.

    Attributes:
        tokenizer (PreTrainedTokenizerBase): Any of huggingface tokenizers that inherits PreTrainedTokenizerBase.
        rate (float): The removal rate.

    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, rate: float = 0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.rate = rate

    def forward(self, inputs):
        """Randomly delete elements in ratio (given rate) from tokens.

        Args:
            inputs (List[List[str]]): A pair of token sequences.
        """

        result = []
        for tokens in inputs:
            tokens = list(tokens)
            n_tokens = len(tokens)
            # to avoid removing the special tokens, we create sample_indices to exclude the special token indices
            sample_indices = np.array([i for i in range(n_tokens) if tokens[i] not in self.tokenizer.all_special_tokens_extended])
            n_removal = int(n_tokens * self.rate)
            removal_indices = np.random.choice(sample_indices, size=n_removal, replace=False)

            result.append([tokens[i] for i in range(n_tokens) if i not in removal_indices])

        return result


# swap two random tokens `n_swap` times
class RandomSwap(torch.nn.Module):
    """The custom module for random swap implementation.

    Args:
        tokenizer (PreTrainedTokenizerBase): Any of huggingface tokenizers that inherits PreTrainedTokenizerBase.
        n_swap (int): The total swap times.

    Attributes:
        tokenizer (PreTrainedTokenizerBase): Any of huggingface tokenizers that inherits PreTrainedTokenizerBase.
        n_swap (int): The total swap times.

    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, n_swap: int = 1):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_swap = n_swap

    def forward(self, inputs):
        """Randomly swap two elements `n_swap` times for each token sequences in the pair.

        Args:
            inputs (List[List[str]]): A pair of token sequences.
        """

        result = []
        for tokens in inputs:
            tokens = list(tokens)  # we don't want to mutate the inputs
            if len(tokens) > 2:
                sample_indices = np.array([i for i in range(len(tokens)) if tokens[i] not in self.tokenizer.all_special_tokens_extended])
                for i in range(self.n_swap):
                    swap_indices = np.random.choice(sample_indices, size=2, replace=False)
                    tokens[swap_indices[0]], tokens[swap_indices[1]] = tokens[swap_indices[1]], tokens[swap_indices[0]]

            result.append(tokens)

        return result


# randomly mask `n_mask` tokens
class RandomMask(torch.nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, rate: float = 0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.rate = rate

    def forward(self, inputs):
        """Randomly mask tokens in ratio (given rate).

        Args:
            inputs (List[List[str]]): A pair of token sequences.
        """

        result = []
        for tokens in inputs:
            tokens = list(tokens)
            n_tokens = len(tokens)
            sample_indices = np.array([i for i in range(n_tokens) if tokens[i] not in self.tokenizer.all_special_tokens_extended])
            n_mask = int(n_tokens * self.rate)
            mask_indices = np.random.choice(sample_indices, size=n_mask, replace=False)
            for idx in mask_indices:
                tokens[idx] = self.tokenizer.mask_token
            result.append(tokens)

        return result
