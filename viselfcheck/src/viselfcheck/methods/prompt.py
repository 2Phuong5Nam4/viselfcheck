import time
import os
from typing import List, Optional
from tqdm import tqdm
import numpy as np
from openai import OpenAI
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Please install it or provide API keys manually.")

from ..base import SelfCheckBase

class SelfCheckAPIPrompt(SelfCheckBase):
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via API-based prompting (e.g., OpenAI's GPT)
    """
    def __init__(
        self,
        client_type: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # Load defaults from environment variables
        self.client_type = client_type or os.getenv('DEFAULT_API_TYPE', 'openai')
        
        # Set model based on client type
        if model is None:
            if self.client_type == "openai":
                self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            elif self.client_type == "groq":
                self.model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
            elif self.client_type == "gemini":
                self.model = os.getenv('GEMINI_MODEL', 'gemini-pro')
            else:
                self.model = 'gpt-3.5-turbo'
        else:
            self.model = model

        # Set API key based on client type
        if api_key is None:
            if self.client_type == "openai":
                api_key = os.getenv('OPENAI_API_KEY')
            elif self.client_type == "groq":
                api_key = os.getenv('GROQ_API_KEY')
            elif self.client_type == "gemini":
                api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError(f"API key not provided for {self.client_type}. "
                           f"Please set it in .env file or pass it as parameter.")

        # Initialize client based on type
        if self.client_type == "openai":
            self.client = OpenAI(api_key=api_key)
            print(f"✅ Initiate OpenAI client... model = {self.model}")
        elif self.client_type == "groq":
            self.client = Groq(api_key=api_key)
            print(f"✅ Initiate Groq client... model = {self.model}")
        elif self.client_type == 'gemini':
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            print(f"✅ Initiate Gemini client... model = {self.model}")
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

        self.prompt_template = "Ngữ cảnh: {context}\n\nCâu: {sentence}\n\nXác định xem câu trên có nhất quán với ngữ cảnh đã cho hay không. Trả lời 'Có' nếu câu phù hợp với ngữ cảnh và không mâu thuẫn với thông tin đã cung cấp. Trả lời 'Không' nếu câu mâu thuẫn hoặc không được hỗ trợ bởi ngữ cảnh.\n\nTrả lời: "
        self.text_mapping = {'có': 0.0, 'không': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()


    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    def completion(self, prompt: str):
        if self.client_type == "openai" or self.client_type == "groq":
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # 0.0 = deterministic,
                max_tokens=5, # max_tokens is the generated one,
            )
            return chat_completion.choices[0].message.content

        else:
            raise ValueError("client_type not implemented")

    def completion_test(self, prompt: str):
        if self.client_type == "openai" or self.client_type == "groq" or self.client_type == 'gemini':
            while True:
                try:
                    chat_completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            # {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0, # 0.0 = deterministic,
                        max_tokens=5, # max_tokens is the generated one,
                    )

                    break

                except Exception as e:
                    print(e)
                    time.sleep(10)

            return chat_completion.choices[0].message.content

        else:
            raise ValueError("client_type not implemented")

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
        **kwargs
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        
        Args:
            sentences: List of sentences to be evaluated, e.g. GPT text response split by spacy
            sampled_passages: List of stochastically generated responses (without sentence splitting)
            verbose: If True, tqdm progress bar will be shown (default: False)
            **kwargs: Additional parameters for future extensibility
            
        Returns:
            List of sentence-level scores (0-1 range, higher means more inconsistent)
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ")
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                # generate_text = self.completion(prompt)
                generate_text = self.completion_test(prompt)
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