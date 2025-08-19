import os
import numpy as np
import py_vncorenlp
import threading

# Global singleton for VnCoreNLP to avoid JVM conflicts
_vncorenlp_instance = None
_vncorenlp_lock = threading.Lock()

# ======================================== Scoring methods for mqag ========================================
def method_simple_counting(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    AT,
):
    """
    simple counting method score => count_mismatch / (count_match + count_mismatch)
    :return score: 'inconsistency' score
    """
    # bad questions, i.e. not answerable given the passage
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_good_sample, count_match = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            count_good_sample += 1
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
    if count_good_sample == 0:
        score = 0.5
    else:
        score = (count_good_sample-count_match) / count_good_sample
    return score

def method_vanilla_bayes(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2, AT,
):
    """
    (vanilla) bayes method score: compute P(sentence is non-factual | count_match, count_mismatch)
    :return score: 'inconsistency' score
    """
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
            else:
                count_mismatch += 1
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def method_bayes_with_alpha(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2,
):
    """
    bayes method (with answerability score, i.e. soft-counting) score
    :return score: 'inconsistency' score
    """
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        ans_score = u_score_s[s]
        a_S = np.argmax(prob_s[s])
        if a_DT == a_S:
            count_match += ans_score
        else:
            count_mismatch += ans_score
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

# ============================== List expansion methods for Bert Score ================================
def expand_list1(mylist, num):
    expanded = []
    for x in mylist:
        for _ in range(num):
            expanded.append(x)
    return expanded

def expand_list2(mylist, num):
    expanded = []
    for _ in range(num):
        for x in mylist:
            expanded.append(x)
    return expanded

# ============================== Word segmentation utilities for NLI ================================
def get_vncorenlp_path():
    """
    Get the standardized VnCoreNLP installation path relative to the package root.
    This ensures VnCoreNLP is always located in the same place regardless of working directory.
    """
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return os.path.join(package_root, "vncorenlp")

def seg_fn(text, rdrsegmenter):
    return ' '.join(rdrsegmenter.word_segment(text))

def seg_list_fn(text_list, rdrsegmenter):
    return [seg_fn(text, rdrsegmenter) for text in text_list]

def get_word_segmentation_model():
    """
    Get VnCoreNLP word segmentation model.
    Uses singleton pattern to avoid JVM conflicts when multiple methods need VnCoreNLP.
    """
    global _vncorenlp_instance
    
    # Double-checked locking pattern for thread safety
    if _vncorenlp_instance is None:
        with _vncorenlp_lock:
            if _vncorenlp_instance is None:
                _vncorenlp_instance = _create_vncorenlp_instance()
    
    return _vncorenlp_instance

def _create_vncorenlp_instance():
    """
    Internal function to create VnCoreNLP instance.
    This handles the actual VnCoreNLP initialization logic.
    VnCoreNLP will always be installed and loaded from a fixed location within the package.
    """
    # Always use a fixed location relative to the package root for VnCoreNLP
    # This ensures VnCoreNLP is always loaded from the same location regardless of working directory
    vncorenlp_path = get_vncorenlp_path()
    
    # Check if VnCoreNLP is already installed at the package location
    if not (os.path.exists(vncorenlp_path) and 
            os.path.exists(os.path.join(vncorenlp_path, "models")) and 
            os.path.exists(os.path.join(vncorenlp_path, "VnCoreNLP-1.2.jar"))):
        # Create directory and download VnCoreNLP if not exists
        print(f"VnCoreNLP not found at {vncorenlp_path}. Downloading...")
        if not os.path.exists(vncorenlp_path):
            os.makedirs(vncorenlp_path)
        py_vncorenlp.download_model(save_dir=vncorenlp_path)
        print(f"VnCoreNLP downloaded to: {vncorenlp_path}")
    else:
        print(f"Using existing VnCoreNLP installation at: {vncorenlp_path}")

    try:
        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir=vncorenlp_path)
        print(f"VnCoreNLP word segmentation model initialized successfully from: {vncorenlp_path}")
        return rdrsegmenter
    except Exception as e:
        # Handle JVM already running error
        if "VM is already running" in str(e):
            print("JVM is already running. Attempting to reuse existing VnCoreNLP instance...")
            # Try to get existing instance through Java reflection
            try:
                # This is a fallback - in practice, we should have caught this at the singleton level
                rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir=vncorenlp_path)
                return rdrsegmenter
            except Exception as inner_e:
                raise RuntimeError(f"Failed to reuse existing VnCoreNLP instance: {inner_e}")
        else:
            raise RuntimeError(f"Error initializing VnCoreNLP from {vncorenlp_path}: {e}")

def reset_vncorenlp_instance():
    """
    Reset the VnCoreNLP singleton instance.
    This is mainly useful for testing or when you need to reinitialize VnCoreNLP.
    """
    global _vncorenlp_instance
    with _vncorenlp_lock:
        _vncorenlp_instance = None
        print("VnCoreNLP instance reset")

def is_vncorenlp_initialized():
    """
    Check if VnCoreNLP instance is already initialized.
    """
    global _vncorenlp_instance
    return _vncorenlp_instance is not None