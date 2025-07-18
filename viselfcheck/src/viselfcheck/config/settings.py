class MQAGConfig:
    qa_generator_check_point: str = "2Phuong5Nam4/VIT5-base-QA-Generation"
    distractor_generator_check_point: str = "2Phuong5Nam4/VIT5-base-Distractors-Generation"
    answerer_check_point: str = "2Phuong5Nam4/xlm-roberta-base-MCQ-Answering"
    answerability: str = "2Phuong5Nam4/xlm-roberta-base-Answerable"
    num_questions_per_sent: int = 5
    scoring_method: str = "bayes_with_alpha"
    AT: float = 0.5
    beta1: float = 0.8
    beta2: float = 0.8


class NLIConfig:
    nli_model: str = "pgnguyen/phobert-large-nli"
    do_word_segmentation: bool = True


class BertScoreConfig:
    model_type: str = "vinai/phobert-base"  # Default model for Vietnamese