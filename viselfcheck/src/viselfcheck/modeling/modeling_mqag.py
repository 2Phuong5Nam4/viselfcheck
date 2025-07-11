import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMultipleChoice, AutoModelForSequenceClassification

class QAGenerator:
    def __init__(self, checkpoint_path=None, device=None, max_length=512):
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        print("QA Generator loaded successfully!")
    
    def prepare_qa_input(
        self,
        contexts,
    ):

        encoding = self.tokenizer(
            contexts,
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids
        input_ids = input_ids.to(self.device)
        return input_ids
    
    def batch_generate(self, contexts):
        qa_input_ids = self.prepare_qa_input(contexts)
        outputs = self.model.generate(
                qa_input_ids,
                max_new_tokens=128,
                do_sample=True,
            )
        
        questions, answers = [], []
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        for inx, decoded_output in enumerate(decoded_outputs):
            question_answer = decoded_output.replace(self.tokenizer.pad_token, "").replace(self.tokenizer.eos_token, "")
            # print("Decoded output:", question_answer)
            question_answer_split = question_answer.split(self.tokenizer.sep_token)
            question, answer = "", ""
            if len(question_answer_split) == 2:
                question = question_answer_split[0].strip()
                answer = question_answer_split[1].strip()

            else:   
                num_retries = 3
                for _ in range(num_retries):
                    retried_output = self.model.generate(qa_input_ids[inx:inx+1], max_new_tokens=128, do_sample=True)
                    retried_decoded_output = self.tokenizer.decode(retried_output[0], skip_special_tokens=False)
                    retried_question_answer = retried_decoded_output.replace(self.tokenizer.pad_token, "").replace(self.tokenizer.eos_token, "")
                    retried_question_answer_split = retried_question_answer.split(self.tokenizer.sep_token)
                    if len(retried_question_answer_split) == 2:
                        question = retried_question_answer_split[0].strip()
                        answer = retried_question_answer_split[1].strip()
                        break
                else:
                    # If all retries fail, set question and answer to empty strings
                    # and print an error message
                    print(f"Error: Question and Answer not generated for context {inx}!")

            
            questions.append(question)
            answers.append(answer)
            
        return questions, answers

    @torch.no_grad()
    def generate(self, contexts, batch_size=16):
        """
        input: context (list of strings)
        output: lists of questions and answers
        """
        all_questions, all_answers = [], []
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            questions, answers = self.batch_generate(batch_contexts)
            all_questions.extend(questions)
            all_answers.extend(answers)
            
        return all_questions, all_answers

class DistractorGenerator:
    def __init__(self, checkpoint_path=None, device=None, max_length=512, num_distractions=3):
        self.max_length = max_length 
        self.num_distractions = num_distractions

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        print("Distractor Generator loaded successfully!")
    
    def prepare_distractor_input(
        self,
        contexts,
        questions,
        answers,
        separator='<sep>',
    ):
        """
        input: question <sep> answer <sep> article
        output: distractor1 <sep> distractor2 <sep> distractor3
        """
        input_texts = []
        for context, question, answer in zip(contexts, questions, answers):
            input_text = question + ' ' + separator + ' ' + answer + ' ' + separator + ' ' + context
            input_texts.append(input_text)

        encoding = self.tokenizer(
            input_texts,
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids
        input_ids = input_ids.to(self.device)
        return input_ids
    
    def batch_generate(self, contexts, questions, answers):
        distractor_input_ids = self.prepare_distractor_input(
            contexts,
            questions,
            answers,
        )
        
        outputs = self.model.generate(
                distractor_input_ids,
                max_new_tokens=128,
                do_sample=True,
            )
        list_of_distractors = []
        decdecoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        for output in decdecoded_outputs:
            distractors = output.replace(self.tokenizer.pad_token, "").replace(self.tokenizer.eos_token, "")
            distractors = [y.strip() for y in distractors.split(self.tokenizer.sep_token)]
            if len(distractors) < self.num_distractions:
                while len(distractors) < self.num_distractions:
                    distractors.append(distractors[-1])
            elif len(distractors) > self.num_distractions:
                distractors = distractors[:self.num_distractions]
            list_of_distractors.append(distractors)
        return list_of_distractors
    
    @torch.no_grad()
    def generate(self, contexts, questions, answers, batch_size=16):
        """
        input: context (list of strings), questions (list of strings), answers (list of strings)
        output: list of lists of distractors
        """
        all_distractors = []
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            batch_questions = questions[i:i+batch_size]
            batch_answers = answers[i:i+batch_size]
            distractors = self.batch_generate(batch_contexts, batch_questions, batch_answers)
            all_distractors.extend(distractors)

        return all_distractors

class QuestionAnswerer:
    def __init__(self, checkpoint_path=None, device=None, max_length=512):
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForMultipleChoice.from_pretrained(checkpoint_path)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        print("Question Answerer loaded successfully!") 

    def prepare_answering_input(
        self,
        contexts,
        questions,
        options,
        num_options,
    ):
        """
        this currently only supports longformer
        """
        c_plus_q_array = []
        option_array = []
        for (context, question, option) in zip(contexts, questions, options):
            assert len(option) == num_options
            c_plus_q = context + ' ' + self.tokenizer.bos_token + ' ' + question
            c_plus_q_array.append([c_plus_q] * num_options)
            option_array.append(option)

        c_plus_q_array = sum(c_plus_q_array, [])
        option_array = sum(option_array, [])

        tokenized_examples = self.tokenizer(
            option_array, c_plus_q_array, 
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            # return_tensors="pt",
        )
        encoding = {k: [v[i : i + num_options] for i in range(0, len(v), num_options)] for k, v in tokenized_examples.items()}
        input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        example_encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return example_encoded
    
    def batch_predict(self, contexts, questions, options):
        num_options = len(options[0])
        inputs = self.prepare_answering_input(
            contexts,
            questions,
            options,
            num_options
        )
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs['logits'], dim=-1).cpu().tolist()
        return probs
    

    @torch.no_grad()
    def predict(self, context, question, options, batch_size=16):
        """
        input: context (list of strings), questions (list of strings), options (list of lists of strings)
        output: list of probabilities
        """
        all_probs = []
        for i in range(0, len(context), batch_size):
            batch_contexts = context[i:i+batch_size]
            batch_questions = question[i:i+batch_size]
            batch_options = options[i:i+batch_size]
            probs = self.batch_predict(batch_contexts, batch_questions, batch_options)
            all_probs.extend(probs)

        return all_probs
    

class QuestionCurator:
    def __init__(self, checkpoint_path=None, device=None, max_length=512):
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=1, problem_type="regression", ignore_mismatched_sizes=True)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        print("Question Curator loaded successfully!")
    def prepare_answering_input(
        self,
        contexts,
        questions,
    ):
        """
        this currently only supports longformer
        """
        input_array = []
        for (context, question) in zip(contexts, questions):
            input = context + ' ' + self.tokenizer.bos_token + ' ' + question
            input_array.append(input)

        tokenized_examples = self.tokenizer(
            input_array,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized_examples['input_ids']
        attention_mask = tokenized_examples['attention_mask'] 

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        example_encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return example_encoded
    
    def batch_predict(self, contexts, questions):
        inputs = self.prepare_answering_input(contexts, questions)
        outputs = self.model(**inputs)
        probs = outputs.logits.squeeze().cpu()
        # clip probs to be between 0 and 1
        probs = torch.clamp(probs, 0, 1)
        if probs.dim() == 0:
            probs = [probs.item()]
        else:
            probs = probs.tolist()
        return probs
    
    @torch.no_grad()
    def predict(self, contexts, questions, batch_size=16):
        """
        input: context (list of strings), questions (list of strings)
        output: list of probabilities
        """
        all_probs = []
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            batch_questions = questions[i:i+batch_size]
            probs = self.batch_predict(batch_contexts, batch_questions)
            all_probs.extend(probs)

        return all_probs