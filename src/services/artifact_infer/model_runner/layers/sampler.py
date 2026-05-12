import torch
from torch import nn

from ...engine.io_struct import SamplingInfo

_SGL_KERNEL_OPS = None
_SGL_KERNEL_IMPORT_ERROR = None


def _load_sgl_kernel_ops():
    global _SGL_KERNEL_OPS, _SGL_KERNEL_IMPORT_ERROR
    if _SGL_KERNEL_OPS is not None:
        return _SGL_KERNEL_OPS
    if _SGL_KERNEL_IMPORT_ERROR is not None:
        return None
    try:
        from sgl_kernel import (
            min_p_sampling_from_probs,
            top_k_renorm_prob,
            top_k_top_p_sampling_from_probs,
            top_p_renorm_prob,
        )
    except ImportError as exc:
        _SGL_KERNEL_IMPORT_ERROR = exc
        return None

    _SGL_KERNEL_OPS = {
        "min_p_sampling_from_probs": min_p_sampling_from_probs,
        "top_k_renorm_prob": top_k_renorm_prob,
        "top_k_top_p_sampling_from_probs": top_k_top_p_sampling_from_probs,
        "top_p_renorm_prob": top_p_renorm_prob,
    }
    return _SGL_KERNEL_OPS


def _apply_sampling_filters(probs: torch.Tensor, sampling_infos: SamplingInfo) -> torch.Tensor:
    filtered = probs.clone()
    vocab_size = filtered.size(-1)

    for row_idx in range(filtered.size(0)):
        row = filtered[row_idx]
        top_k = int(sampling_infos.top_ks[row_idx].item())
        top_p = float(sampling_infos.top_ps[row_idx].item())
        min_p = float(sampling_infos.min_ps[row_idx].item())

        if 0 < top_k < vocab_size:
            threshold = torch.topk(row, top_k).values[-1]
            row.masked_fill_(row < threshold, 0)

        if 0 < top_p < 1:
            sorted_probs, sorted_indices = torch.sort(row, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[1:] = sorted_remove[:-1].clone()
            sorted_remove[0] = False
            row.scatter_(0, sorted_indices[sorted_remove], 0)

        if min_p > 0:
            row.masked_fill_(row < row.max() * min_p, 0)

        prob_sum = row.sum()
        if prob_sum > 0:
            row.div_(prob_sum)
        else:
            row.zero_()
            row[torch.argmax(probs[row_idx])] = 1

    return filtered


def _torch_sample_from_probs(probs: torch.Tensor, sampling_infos: SamplingInfo) -> torch.Tensor:
    probs = _apply_sampling_filters(probs, sampling_infos)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    # def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    #     logits = logits.to(torch.float)
    #     greedy_tokens = logits.argmax(dim=-1)
    #     logits.div_(temperatures.unsqueeze(dim=1))
    #     probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    #     # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
    #     epsilon = 1e-10  
    #     sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
    #     return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
    
    def forward(self, logits: torch.Tensor, sampling_infos: SamplingInfo):
        greedy_tokens = logits.argmax(dim=-1)
        if sampling_infos.is_greedy_sampling:
            return greedy_tokens
        greedy_mask = sampling_infos.temperatures < 0
        temperatures = sampling_infos.temperatures.masked_fill(greedy_mask, 1.0)
        logits = logits.float().div_(temperatures[:, None])
        probs = torch.softmax(logits, dim=-1)
        del logits

        sgl_ops = _load_sgl_kernel_ops()
        if sgl_ops is None:
            sample_tokens = _torch_sample_from_probs(probs, sampling_infos)
            return torch.where(greedy_mask, greedy_tokens, sample_tokens)

        if sampling_infos.need_min_p_sampling:
            probs = sgl_ops["top_k_renorm_prob"](probs, sampling_infos.top_ks)
            probs = sgl_ops["top_p_renorm_prob"](probs, sampling_infos.top_ps)
            sample_tokens = sgl_ops["min_p_sampling_from_probs"](probs, sampling_infos.min_ps)
        else:
            sample_tokens = sgl_ops["top_k_top_p_sampling_from_probs"](
                probs.contiguous(),
                sampling_infos.top_ks,
                sampling_infos.top_ps,
                filter_apply_order="joint",
            )
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)
