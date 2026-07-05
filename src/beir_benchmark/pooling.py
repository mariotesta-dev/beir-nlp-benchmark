import torch

def mean_pooling(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()     # (B, L) -> (B, L, 1)
    summed = (last_hidden_states * mask).sum(dim=1) # (B, L, D) * (B, L, 1) -> (B, D)
    counts = mask.sum(dim=1).clamp(min=1e-9)        # (B, L, 1) -> (B, 1)
    return summed / counts                          # (B, D) / (B, 1) -> (B, D)

def cls_pooling(last_hidden_states, attention_mask):
    return last_hidden_states[:, 0]                 # (B, L, D) -> (B, D)

def last_token_pool(last_hidden_state, attention_mask):
    left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padded:
        return last_hidden_state[:, -1]
    
    lengths = attention_mask.sum(dim=1) - 1
    B = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(B), lengths] # (B, L, D) -> (B, D)


POOLING_FUNCTIONS = {
    "mean": mean_pooling,
    "cls": cls_pooling,
    "last_token": last_token_pool,
}