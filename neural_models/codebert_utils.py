from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

MODEL_NAME = "microsoft/codebert-base"
EMBEDDING_DIM = 768

def load_codebert_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def get_codebert_embedding(text, tokenizer, model, device=None):
    """
    Returns the [CLS] embedding (numpy array) for the given text using CodeBERT.
    If text is None or empty, returns a zero vector.
    """
    if not text:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # [CLS] token is at position 0
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_emb.astype(np.float32)
