from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    
    def __init__(self, alphabet: List[str] = None):
        self.EMPTY_TOK = "^"
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.EMPTY_IND = self.char2ind[self.EMPTY_TOK]

    def ctc_decode(self, inds: List[int]) -> str:
        last = self.EMPTY_IND
        result = []
        for ind in inds:
            if ind != last:
                last = ind
                if ind != self.EMPTY_IND:
                    result.append(self.ind2char[ind])
        return ''.join(result)


    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        # TODO: your code here
        raise NotImplementedError
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
