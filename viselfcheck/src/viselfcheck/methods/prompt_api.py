from typing import List, Optional
from tqdm import tqdm

import numpy as np

from openai import OpenAI

from ..config.settings import APIPromptConfig
from ..base import SelfCheckBase


class SelfCheckAPIPrompt(SelfCheckBase):
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via API-based prompting
    """
    def __init__(
        self,
        model = None,
        base_url = None,
        api_key = None,
    ):
        self.model = model if model is not None else APIPromptConfig.llm_model
        self.base_url = base_url if base_url is not None else APIPromptConfig.base_url
        self.prompt_template = "Ngữ cảnh: {context}\n\nCâu: {sentence}\n\nXác định xem câu trên có nhất quán với ngữ cảnh đã cho hay không. Trả lời 'Có' nếu câu phù hợp với ngữ cảnh và không mâu thuẫn với thông tin đã cung cấp. Trả lời 'Không' nếu câu mâu thuẫn hoặc không được hỗ trợ bởi ngữ cảnh.\n\nTrả lời: "
        self.text_mapping = {'có': 0.0, 'không': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key
        )

        print("SelfCheckPrompt initialized")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    def completion(self, prompt: str):
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5,
        )

        return chat_completion.choices[0].message.content

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        passage: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose

        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]

            for sample_i, sample in enumerate(sampled_passages):
                sample = sample.replace("\n", " ")
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                generate_text = self.completion(prompt)

                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_

        scores_per_sentence = scores.mean(axis=-1)

        return scores_per_sentence.tolist()

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        text = text.lower().strip()
        if text[:2] == 'có':
            text = 'có'
        elif text[:5] == 'không':
            text = 'không'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'

        return self.text_mapping[text]