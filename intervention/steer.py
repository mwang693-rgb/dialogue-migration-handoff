"""Additive residual hook, greedy generation and teacher-forced KL for one decoder block.

Positions follow intervention/protocol.json: the last prompt token and every later
token. ``Steer.start`` is the first in-call position to modify, so the same hook
serves a prefill call (start = prompt_len - 1), a cached decode call (start = 0) and
a teacher-forced pass over prompt + reference (start = prompt_len - 1).
"""
import torch


class Steer:
    def __init__(self, model, layer, vector=None):
        self.block = model.model.layers[layer - 1]  # layer k = output of block k
        self.vector = vector
        self.start = None
        self.calls = []
        self.handle = None

    def __enter__(self):
        if self.vector is not None:
            self.handle = self.block.register_forward_hook(self.hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()

    def hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        assert h.dim() == 3 and h.shape[0] == 1 and self.start is not None and 0 <= self.start < h.shape[1]
        z = h.clone()
        v = self.vector.to(device=h.device, dtype=torch.float32)
        z[:, self.start:, :] = (h[:, self.start:, :].float() + v).to(h.dtype)
        self.calls.append((h.shape[1], self.start))
        self.start = None  # each call must set its own scope
        return (z,) + tuple(output[1:]) if isinstance(output, tuple) else z


@torch.inference_mode()
def generate(model, ids, eos, cap, steer):
    """Greedy decode with the confirmation call pattern: full prefill, then one token per call."""
    dev = model.device
    steer.start = len(ids) - 1
    out = model(input_ids=torch.tensor([ids], device=dev), attention_mask=torch.ones((1, len(ids)), device=dev, dtype=torch.long),
                use_cache=True, logits_to_keep=1)
    logits, cache, generated = out.logits[0, -1], out.past_key_values, []
    for i in range(cap):
        token = int(logits.argmax())
        generated.append(token)
        if token in eos or i == cap - 1:
            break
        steer.start = 0
        n = cache.get_seq_length()
        out = model(input_ids=torch.tensor([[token]], device=dev), attention_mask=torch.ones((1, n + 1), device=dev, dtype=torch.long),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits, cache = out.logits[0, -1], out.past_key_values
    del cache
    natural = generated[-1] in eos
    return dict(generated_ids=generated, token_ids=generated[:-1] if natural else generated, truncated=not natural)


@torch.inference_mode()
def teacher_forced_logprobs(model, ids, reference, steer):
    """Log-probabilities (float32) predicting each reference token after the prompt."""
    full = ids + reference[:-1]
    dev = model.device
    steer.start = len(ids) - 1
    out = model(input_ids=torch.tensor([full], device=dev), attention_mask=torch.ones((1, len(full)), device=dev, dtype=torch.long),
                use_cache=False, logits_to_keep=len(reference))
    logp = torch.log_softmax(out.logits[0].float(), dim=-1)
    assert logp.shape[0] == len(reference)
    return logp


def kl_summary(ref_logp, logp, reference):
    """Per-position KL(ref || other); mean, first position, top-1 agreement, reference-token log-prob."""
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1).clamp_min(0)
    target = torch.tensor(reference, device=logp.device)
    return dict(kl_mean=float(kl.mean()), kl_first=float(kl[0]), kl_max=float(kl.max()),
                top1_agreement=float((ref_logp.argmax(-1) == logp.argmax(-1)).float().mean()),
                ref_token_logprob_mean=float(logp.gather(1, target[:, None]).mean()), positions=len(reference))
