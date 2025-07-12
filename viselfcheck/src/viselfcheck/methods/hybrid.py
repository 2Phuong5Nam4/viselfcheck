import os
import time
import numpy as np
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

from ..utils.utils import seg_list_fn, get_word_segmentation_model
from ..config.settings import NLIConfig
from ..base import SelfCheckBase

class SelfCheckHybrid(SelfCheckBase):
    def __init__(
        self,
        nli_model: Optional[str] = None,
        device = None,
        do_word_segmentation = None,
        llm_model: Optional[str] = None,
        api_key = None
    ):
        # NLI intit
        if nli_model is None:
            nli_model = NLIConfig.nli_model
        self.nli_model = nli_model
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.do_word_segmentation = do_word_segmentation if do_word_segmentation is not None else NLIConfig.do_word_segmentation

        self.model = AutoModelForSequenceClassification.from_pretrained(self.nli_model).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model)

        if self.do_word_segmentation:
            self.rdrsegmenter = get_word_segmentation_model()

        # API LLM init
        self.client = OpenAI(
            # base_url='https://api.together.xyz/inference',
            base_url='https://generativelanguage.googleapis.com/v1beta/openai',
            api_key=api_key
        )

        if llm_model is None:
            llm_model = 'gpt-3.5-turbo'
        self.llm_model = llm_model
        self.prompt_template = "Ngữ cảnh: {context}\n\nCâu: {sentence}\n\nXác định xem câu trên có nhất quán với ngữ cảnh đã cho hay không. Trả lời 'Có' nếu câu phù hợp với ngữ cảnh và không mâu thuẫn với thông tin đã cung cấp. Trả lời 'Không' nếu câu mâu thuẫn hoặc không được hỗ trợ bởi ngữ cảnh.\n\nTrả lời: "
        self.text_mapping = {'có': 0.0, 'không': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()

        print('SelfCheckHybrid initialized')

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        **kwargs
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        
        Args:
            sentences: List of sentences to be evaluated, e.g. GPT text response split by spacy
            sampled_passages: List of stochastically generated responses (without sentence splitting)
            **kwargs: Additional parameters for future extensibility
            
        Returns:
            List of sentence-level scores (0-1 range, higher means more inconsistent)
            Note: Combines NLI and LLM prompting - uses LLM when NLI is uncertain
        """
        if self.do_word_segmentation:
            sentences_ = seg_list_fn(sentences, self.rdrsegmenter)
            sampled_passages_ = seg_list_fn(sampled_passages, self.rdrsegmenter)
        else:
            sentences_ = sentences
            sampled_passages_ = sampled_passages

        assert len(sentences) == len(sentences_)
        assert len(sampled_passages) == len(sampled_passages_)

        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))

        for i in range(num_sentences):
            for j in range(num_samples):
                inputs = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=[(sentences_[i], sampled_passages_[j])],
                    return_token_type_ids=True, return_attention_mask=True,
                    add_special_tokens=True, padding='longest',
                    truncation=True, return_tensors='pt',
                    max_length=256
                )

                inputs = inputs.to(self.device)

                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                prob_e = probs[0][0]
                prob_n = probs[0][1]
                prob_c = probs[0][2]

                if prob_e < prob_n + prob_c:
                    prompt = self.prompt_template.format(context=sampled_passages[j], sentence=sentences[i])
                    llm_score = self.llm_predict(prompt)
                    scores[i, j] = llm_score
                else:
                    scores[i, j] = 0.0

        scores_per_sentence = scores.mean(axis=-1)

        return scores_per_sentence.tolist()

    def llm_predict(self, prompt: str):
        while True:
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=0.0,
                    max_tokens=5,
                )
                break

            except Exception as e:
                time.sleep(5)

        return self.text_postprocessing(chat_completion.choices[0].message.content)

    def text_postprocessing(
        self,
        text,
    ):
        text = text.lower().strip()
        if text[:2] == 'có':
            text = 'có'
        elif text[:5] == 'không':
            text = 'không'
        else:
            if text not in self.not_defined_text:
                print(f'warning: {text} not defined')
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]