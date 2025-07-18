import torch
from typing import List, Optional
import torch
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import random


from ..modeling.modeling_mqag import (QAGenerator, 
                                      DistractorGenerator, 
                                      QuestionAnswerer, 
                                      QuestionCurator)
from ..config.settings import MQAGConfig
from ..utils.utils import method_simple_counting, method_vanilla_bayes, method_bayes_with_alpha


from ..base import SelfCheckBase

class SelfCheckMQAG(SelfCheckBase):
    """
    SelfCheckGPT (MQAG varaint): Checking LLM's text against its own sampled texts via MultipleChoice Question Answering
    """
    def __init__(
        self,
        QAGenerator_checkpoint: Optional[str] = None,
        DistractorGenerator_checkpoint: Optional[str] = None,
        QuestionAnswerer_checkpoint: Optional[str] = None,
        QuestionCurator_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
        num_questions_per_sent: Optional[int] = None,
        scoring_method: Optional[str] = None,
        AT: Optional[float] = None,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
    ):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("SelfCheck-MQAG initialized to device", self.device)

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if QAGenerator_checkpoint is None:
            QAGenerator_checkpoint = MQAGConfig.qa_generator_check_point
        if DistractorGenerator_checkpoint is None:
            DistractorGenerator_checkpoint = MQAGConfig.distractor_generator_check_point
        if QuestionAnswerer_checkpoint is None:
            QuestionAnswerer_checkpoint = MQAGConfig.answerer_check_point
        if QuestionCurator_checkpoint is None:
            QuestionCurator_checkpoint = MQAGConfig.answerability

        self.qa_generator = QAGenerator(QAGenerator_checkpoint, device=self.device)
        self.distractor_generator = DistractorGenerator(DistractorGenerator_checkpoint, device=self.device)
        self.answerer = QuestionAnswerer(QuestionAnswerer_checkpoint, device=self.device)
        self.curator = QuestionCurator(QuestionCurator_checkpoint, device=self.device)
        
        # Set default parameters
        self.num_questions_per_sent = num_questions_per_sent if num_questions_per_sent is not None else MQAGConfig.num_questions_per_sent
        self.scoring_method = scoring_method if scoring_method is not None else MQAGConfig.scoring_method
        self.AT = AT if AT is not None else MQAGConfig.AT
        self.beta1 = beta1 if beta1 is not None else MQAGConfig.beta1
        self.beta2 = beta2 if beta2 is not None else MQAGConfig.beta2


    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        passage: Optional[str] = None,
        **kwargs
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        
        Args:
            sentences: List of sentences to be evaluated, e.g. GPT text response split by spacy
            sampled_passages: List of stochastically generated responses (without sentence splitting)
            passage: Optional passage text. If provided, this will be used instead of joining sentences.
                    If None, will use " ".join(sentences) to create the passage.
            **kwargs: Additional parameters for future extensibility
            
        Returns:
            List of sentence-level scores (inconsistency scores, higher means likely hallucination)
        """
        # Use instance variables for parameters
        num_questions_per_sent = self.num_questions_per_sent
        scoring_method = self.scoring_method
        AT = self.AT
        beta1 = self.beta1
        beta2 = self.beta2

        assert scoring_method in ['counting', 'bayes', 'bayes_with_alpha']
        num_samples = len(sampled_passages)
        
        # Calculate total steps: For each sentence we have:
        # - num_questions_per_sent question generations
        # - num_questions_per_sent answer checks for main passage
        # - num_questions_per_sent * num_samples answer checks for sampled passages
        
        # Configure progress bar based on environment
        # Use provided passage or join sentences to form a passage
        if passage is not None:
            main_passage = passage
        else:
            main_passage = " ".join(sentences)
        
        extended_passages = [main_passage]*num_questions_per_sent
        extended_sampled_passages = sampled_passages*num_questions_per_sent

        scores = []
        for sentence in sentences:
            sent_scores = []
            # State1: Question + Choices Generation
            extended_sentences = [sentence]*num_questions_per_sent
            questions, answers = self.qa_generator.generate(extended_sentences)
            distractors = self.distractor_generator.generate(extended_passages, questions, answers)
            options = [[answer] + distractor for answer, distractor in zip(answers, distractors)]
            # Answering

            # max_seq_length = 4096 # answering & answerability max length
            # for question_item in questions_answers:
                # response
            probs = self.answerer.predict(extended_passages, questions, options) # (num_questions_per_sent, 4)

            u_score = self.curator.predict(extended_passages, questions) # (num_questions_per_sent,)

            extended_questions = [] # (num_questions_per_sent*num_samples,)
            extened_options = [] # (num_questions_per_sent*num_samples, 4)
            for question, option in zip(questions, options):
                extended_questions.extend([question]*num_samples) 
                extened_options.extend([option]*num_samples)
            prob_s = self.answerer.predict(extended_sampled_passages, extended_questions, extened_options, batch_size=16) # (num_questions_per_sent*num_samples, 4)
            u_score_s = self.curator.predict(extended_sampled_passages, extended_questions, batch_size=16) # (num_questions_per_sent*num_samples,)

            # convert to numpy
            probs = np.array(probs)
            u_score = np.array(u_score)
            prob_s = np.array(prob_s)
            u_score_s = np.array(u_score_s)
            
            # reshape prob_s to (num_questions_per_sent, num_samples, 4)
            prob_s = prob_s.reshape(num_questions_per_sent, num_samples, 4)
            u_score_s = u_score_s.reshape(num_questions_per_sent, num_samples)
            
            for p, u, p_s, u_s in zip(probs, u_score, prob_s, u_score_s):
                score = 0.0

                if scoring_method == 'counting':
                    score = method_simple_counting(p, u, p_s, u_s, num_samples, AT=AT)
                elif scoring_method == 'bayes':
                    score = method_vanilla_bayes(p, u, p_s, u_s, num_samples, beta1=beta1, beta2=beta2, AT=AT)
                elif scoring_method == 'bayes_with_alpha':
                    score = method_bayes_with_alpha(p, u, p_s, u_s, num_samples, beta1=beta1, beta2=beta2)

                sent_scores.append(score)
            scores.append(sent_scores)   

        sent_scores = np.array(scores) # (num_sentences, num_questions_per_sent)
        sent_scores = sent_scores.mean(axis=1) # (num_sentences,)
        
        return sent_scores.tolist()  # Convert to list for consistency with other methods