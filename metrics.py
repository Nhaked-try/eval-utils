def normalize_answer(s):
    """Lower text and remove punctuation, articles, extra whitespace."""
    import re
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = ' '.join(s.split())
    return s

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def bleu_1gram(prediction, reference):
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    ref_set = set(ref_tokens)
    
    matches = sum(1 for t in pred_tokens if t in ref_set)
    return matches / len(pred_tokens) if pred_tokens else 0.0
