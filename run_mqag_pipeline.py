#!/usr/bin/env python3
"""
Practical script to run all four models from modeling_mqag.py
Configuration is automatically loaded from viselfcheck.config.settings
You can also override model paths by setting environment variables:
- MQAG_QA_GENERATOR_PATH
- MQAG_DISTRACTOR_GENERATOR_PATH  
- MQAG_QUESTION_ANSWERER_PATH
- MQAG_QUESTION_CURATOR_PATH
"""

import sys
import os
import torch
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "viselfcheck" / "src"
sys.path.insert(0, str(src_path))

try:
    from viselfcheck.modeling.modeling_mqag import (
        QAGenerator, 
        DistractorGenerator, 
        QuestionAnswerer, 
        QuestionCurator
    )
    from viselfcheck.config.settings import MQAGConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the viselfcheck package is properly installed or the path is correct.")
    sys.exit(1)

# Load configuration from settings
config = MQAGConfig()
MODEL_PATHS = {
    'qa_generator': os.getenv('MQAG_QA_GENERATOR_PATH', config.qa_generator_check_point),
    'distractor_generator': os.getenv('MQAG_DISTRACTOR_GENERATOR_PATH', config.distractor_generator_check_point),
    'question_answerer': os.getenv('MQAG_QUESTION_ANSWERER_PATH', config.answerer_check_point),
    'question_curator': os.getenv('MQAG_QUESTION_CURATOR_PATH', config.answerability),
}

# Additional configuration from settings
PIPELINE_CONFIG = {
    'num_questions_per_sent': config.num_questions_per_sent,
    'scoring_method': config.scoring_method,
    'AT': config.AT,
    'beta1': config.beta1,
    'beta2': config.beta2,
}

def print_results(contexts, questions, answers, distractors, answer_probs, quality_scores):
    """Print formatted results from all models"""
    
    print("\n" + "=" * 100)
    print("COMPLETE RESULTS FROM ALL FOUR MODELS")
    print("=" * 100)
    
    for i in range(len(contexts)):
        print(f"\n{'-' * 50} CONTEXT {i+1} {'-' * 50}")
        print(f"📄 CONTEXT:")
        print(f"   {contexts[i]}")
        
        print(f"\n❓ GENERATED QUESTION:")
        print(f"   {questions[i]}")
        
        print(f"\n✅ GENERATED ANSWER:")
        print(f"   {answers[i]}")
        
        print(f"\n🎯 GENERATED DISTRACTORS:")
        for j, distractor in enumerate(distractors[i], 1):
            print(f"   {j}. {distractor}")
        
        print(f"\n📊 ANSWER PROBABILITIES:")
        options = [answers[i]] + distractors[i][:3]  # Correct answer + 3 distractors
        for j, (option, prob) in enumerate(zip(options, answer_probs[i])):
            indicator = "🎯" if j == 0 else "❌"
            print(f"   {indicator} {option}: {prob:.4f}")
        
        print(f"\n⭐ QUESTION QUALITY SCORE:")
        print(f"   {quality_scores[i]:.4f} (0.0 = poor, 1.0 = excellent)")

def run_complete_pipeline(contexts, model_paths, batch_size=2):
    """
    Run the complete MQAG pipeline with all four models
    
    Args:
        contexts: List of text contexts
        model_paths: Dictionary with paths to all four models
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with all results
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    results = {}
    
    # Validate model paths - check if they are valid strings
    for model_name, path in model_paths.items():
        if not path or not isinstance(path, str):
            raise ValueError(f"Invalid model path for {model_name}: {path}")
    
    try:
        # Step 1: Generate questions and answers
        print("\n🔄 Step 1: Loading QA Generator...")
        qa_generator = QAGenerator(
            checkpoint_path=model_paths['qa_generator'],
            device=device,
            max_length=512
        )
        
        print("   Generating questions and answers...")
        questions, answers = qa_generator.generate(contexts, batch_size=batch_size)
        results['questions'] = questions
        results['answers'] = answers
        print(f"   ✅ Generated {len(questions)} question-answer pairs")
        
        # Step 2: Generate distractors
        print("\n🔄 Step 2: Loading Distractor Generator...")
        distractor_generator = DistractorGenerator(
            checkpoint_path=model_paths['distractor_generator'],
            device=device,
            max_length=512,
            num_distractions=3
        )
        
        print("   Generating distractors...")
        distractors = distractor_generator.generate(contexts, questions, answers, batch_size=batch_size)
        results['distractors'] = distractors
        print(f"   ✅ Generated distractors for {len(distractors)} questions")
        
        # Step 3: Evaluate question answering
        print("\n🔄 Step 3: Loading Question Answerer...")
        question_answerer = QuestionAnswerer(
            checkpoint_path=model_paths['question_answerer'],
            device=device,
            max_length=512
        )
        
        print("   Evaluating answer probabilities...")
        options = [[answer] + distractor_list[:3] for answer, distractor_list in zip(answers, distractors)]
        answer_probs = question_answerer.predict(contexts, questions, options, batch_size=batch_size)
        results['answer_probabilities'] = answer_probs
        print(f"   ✅ Computed answer probabilities for {len(answer_probs)} questions")
        
        # Step 4: Evaluate question quality
        print("\n🔄 Step 4: Loading Question Curator...")
        question_curator = QuestionCurator(
            checkpoint_path=model_paths['question_curator'],
            device=device,
            max_length=512
        )
        
        print("   Evaluating question quality...")
        quality_scores = question_curator.predict(contexts, questions, batch_size=batch_size)
        results['quality_scores'] = quality_scores
        print(f"   ✅ Computed quality scores for {len(quality_scores)} questions")
        
        results['contexts'] = contexts
        return results
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        raise

def display_configuration():
    """Display the loaded configuration and sources"""
    print(f"\n📋 Loaded Model Configuration:")
    
    env_overrides = []
    for model_name, model_path in MODEL_PATHS.items():
        env_var_map = {
            'qa_generator': 'MQAG_QA_GENERATOR_PATH',
            'distractor_generator': 'MQAG_DISTRACTOR_GENERATOR_PATH', 
            'question_answerer': 'MQAG_QUESTION_ANSWERER_PATH',
            'question_curator': 'MQAG_QUESTION_CURATOR_PATH'
        }
        
        env_var = env_var_map.get(model_name)
        if env_var:
            is_override = os.getenv(env_var) is not None
            
            if is_override:
                env_overrides.append(env_var)
                print(f"   {model_name}: {model_path} ⚡ (from {env_var})")
            else:
                print(f"   {model_name}: {model_path}")
        else:
            print(f"   {model_name}: {model_path}")
    
    print(f"\n⚙️  Pipeline Configuration:")
    for param_name, param_value in PIPELINE_CONFIG.items():
        print(f"   {param_name}: {param_value}")
    
    if env_overrides:
        print(f"\n🔧 Environment variable overrides detected: {', '.join(env_overrides)}")

def main():
    # Sample contexts
    contexts = [
        "Hà Nội, thủ đô của Việt Nam, nằm ở vị trí 21°01′B 105°52′Đ﻿ / ﻿21.017°B 105.867°Đ﻿ / 21.017; 105.867.  Thành phố nằm ở phía Bắc Việt Nam, thuộc vùng đồng bằng sông Hồng, trên lưu vực sông Hồng và sông Đuống.  Vị trí địa lý này đặt Hà Nội ở vùng khí hậu nhiệt đới gió mùa, với mùa hè nóng ẩm và mùa đông lạnh khô.  Thành phố có diện tích khoảng 3358,5 km², trải dài trên nhiều huyện, thị xã và quận.  Hà Nội nằm cách biển Đông khoảng 100 km về phía Đông, và có biên giới giáp với các tỉnh Vĩnh Phúc, Bắc Ninh, Hưng Yên, Hà Nam.  Vị trí trung tâm của Hà Nội so với các thành phố lớn khác trong khu vực cũng đóng vai trò quan trọng trong việc kết nối giao thông và kinh tế.","['Hà Nội, thủ đô của Việt Nam, nằm ở vị trí 21°01′B 105°52′Đ\ufeff / \ufeff21.017°B 105.867°Đ\ufeff / 21.017; 105.867.', 'Thành phố nằm ở phía Bắc Việt Nam, thuộc vùng đồng bằng sông Hồng, trên lưu vực sông Hồng và sông Đuống.', 'Vị trí địa lý này đặt Hà Nội ở vùng khí hậu nhiệt đới gió mùa, với mùa hè nóng ẩm và mùa đông lạnh khô.', 'Thành phố có diện tích khoảng 3358,5 km², trải dài trên nhiều huyện, thị xã và quận.', 'Hà Nội nằm cách biển Đông khoảng 100 km về phía Đông, và có biên giới giáp với các tỉnh Vĩnh Phúc, Bắc Ninh, Hưng Yên, Hà Nam.', 'Vị trí trung tâm của Hà Nội so với các thành phố lớn khác trong khu vực cũng đóng vai trò quan trọng trong việc kết nối giao thông và kinh tế.",
        
        "Trần Hưng Đạo (1228-1300), tên thật là Trần Quốc Tuấn, là một vị tướng quân, nhà chính trị lỗi lạc của Việt Nam thời Trần. Ông là người có công lao to lớn trong ba lần kháng chiến chống quân Nguyên Mông thế kỷ XIII, được coi là vị anh hùng dân tộc, người anh hùng bất tử của Việt Nam.  Với tài năng quân sự xuất chúng, ông đã ba lần đánh bại quân Nguyên hùng mạnh, bảo vệ nền độc lập dân tộc.  Không chỉ là một danh tướng, Trần Hưng Đạo còn là một nhà chính trị, chiến lược gia tài ba, với những chiến lược, kế sách độc đáo, sáng tạo, được thể hiện rõ nét trong ""Binh thư yếu lược"" - tác phẩm quân sự nổi tiếng của ông.  Sự nghiệp hiển hách của ông đã để lại dấu ấn sâu đậm trong lịch sử Việt Nam, trở thành biểu tượng của lòng yêu nước, tinh thần quật cường và trí tuệ sáng suốt của dân tộc.","['Trần Hưng Đạo (1228-1300), tên thật là Trần Quốc Tuấn, là một vị tướng quân, nhà chính trị lỗi lạc của Việt Nam thời Trần.', 'Ông là người có công lao to lớn trong ba lần kháng chiến chống quân Nguyên Mông thế kỷ XIII, được coi là vị anh hùng dân tộc, người anh hùng bất tử của Việt Nam.', 'Với tài năng quân sự xuất chúng, ông đã ba lần đánh bại quân Nguyên hùng mạnh, bảo vệ nền độc lập dân tộc.', 'Không chỉ là một danh tướng, Trần Hưng Đạo còn là một nhà chính trị, chiến lược gia tài ba, với những chiến lược, kế sách độc đáo, sáng tạo, được thể hiện rõ nét trong ""Binh thư yếu lược"" - tác phẩm quân sự nổi tiếng của ông.', 'Sự nghiệp hiển hách của ông đã để lại dấu ấn sâu đậm trong lịch sử Việt Nam, trở thành biểu tượng của lòng yêu nước, tinh thần quật cường và trí tuệ sáng suốt của dân tộc."
    ]
    
    print("🚀 MQAG Pipeline Test Script")
    print("=" * 50)
    print(f"📝 Processing {len(contexts)} contexts")
    
    # Display loaded configuration
    display_configuration()
    
    try:
        # Run the complete pipeline
        results = run_complete_pipeline(contexts, MODEL_PATHS, batch_size=2)
        
        # Print results
        print_results(
            results['contexts'],
            results['questions'],
            results['answers'],
            results['distractors'],
            results['answer_probabilities'],
            results['quality_scores']
        )
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📊 Summary: Processed {len(contexts)} contexts and generated {len(results['questions'])} questions")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("Please check your model paths and ensure all models are accessible.")

if __name__ == "__main__":
    main()
