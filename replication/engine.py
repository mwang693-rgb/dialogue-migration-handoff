"""Model-agnostic pieces for the multi-model replication: chat serialization, greedy
generation with explicit positions, last-token state capture, and memory-safe
teacher-forced KL. The additive hook is intervention/steer.py:Steer, unchanged."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'intervention'))

SYSTEM = 'You are a helpful assistant.'
SEPARATORS = {'core': '\n\n--- Current user request ---\n', 'pilot': '\n\nLatest user message:\n'}
FAMILIES = {
    # stop = tokens that end a reply (the family's eos list plus the next-turn opener); turn_end = a clean end of turn
    'qwen25': dict(system=True, attn='sdpa', stop=['<|im_end|>', '<|endoftext|>', '<|im_start|>'], turn_end='<|im_end|>', max_context=32768),
    'llama31': dict(system=True, attn='sdpa', stop=['<|eot_id|>', '<|end_of_text|>', '<|eom_id|>', '<|start_header_id|>'], turn_end='<|eot_id|>',
                    max_context=131072),
    'gemma2': dict(system=False, attn='eager', stop=['<end_of_turn>', '<eos>', '<start_of_turn>'], turn_end='<end_of_turn>', max_context=8192),
}


def depth_layers(n_layers):
    """Eight capture depths k/7 of the stack (k = 0..7); Qwen's 0,4,...,28 for 28 layers."""
    return [round(n_layers * k / 7) for k in range(8)]


def messages(history, query, form, system=True, serializer='core'):
    """B = native role turns; C = the history pasted into one user message ('User:'/'Assistant:' labels);
    C_qa / C_json = the cross-format probes of intervention v1."""
    head = [{'role': 'system', 'content': SYSTEM}] if system else []
    if form == 'B':
        return head + list(history) + [{'role': 'user', 'content': query}]
    if form in ('C', 'C_qa'):
        label = {'C': ('User', 'Assistant'), 'C_qa': ('Q', 'A')}[form]
        body = '\n\n'.join(label[m['role'] != 'user'] + ':\n' + m['content'] for m in history) + SEPARATORS[serializer] + query
    elif form == 'C_json':
        body = json.dumps([{'role': m['role'], 'content': m['content']} for m in history], ensure_ascii=False, indent=1)
        body += '\n\nCurrent user request:\n' + query
    else:
        raise ValueError(form)
    return head + [{'role': 'user', 'content': body}]


class Engine:
    def __init__(self, model_path, family, device=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.family = family
        self.spec = FAMILIES[family]
        self.tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = dtype or (torch.bfloat16 if self.device == 'cuda' else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype, attn_implementation=self.spec['attn'],
                                                          local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden = self.model.config.hidden_size
        ids = lambda names: {self.tok.convert_tokens_to_ids(t) for t in names}
        self.stop = ids(self.spec['stop'])
        gc = self.model.generation_config.eos_token_id
        self.stop |= ({gc} if isinstance(gc, int) else set(gc or []))
        assert self.tok.unk_token_id is None or self.tok.unk_token_id not in self.stop
        self.turn_end = self.tok.convert_tokens_to_ids(self.spec['turn_end'])
        self.special = set(self.tok.all_special_ids) | self.stop

    def encode(self, msgs):
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False)['input_ids']

    def prompt(self, history, query, form, serializer='core'):
        return self.encode(messages(history, query, form, self.spec['system'], serializer))

    def _forward(self, ids, start, cache, use_cache, keep):
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        pos = torch.arange(start, start + len(ids), device=self.device)
        return self.model(input_ids=x, attention_mask=torch.ones((1, start + len(ids)), device=self.device, dtype=torch.long),
                          position_ids=pos[None], cache_position=pos, past_key_values=cache, use_cache=use_cache, logits_to_keep=keep)

    def capture_hooks(self, layers, store):
        """Record the last position's output of block k (layer 0 = embedding output) on the next forward call."""
        handles = []
        for layer in layers:
            module = self.model.model.embed_tokens if layer == 0 else self.model.model.layers[layer - 1]

            def hook(mod, args, out, layer=layer):
                if layer not in store:
                    h = out[0] if isinstance(out, tuple) else out
                    store[layer] = h[0, -1].detach().float().cpu().clone()
            handles.append(module.register_forward_hook(hook))
        return handles

    def generate(self, ids, cap, steer=None, capture=None):
        """Greedy decoding: full prefill (hook scope = last prompt token), then one token per call (scope 0).
        Positions are passed explicitly, so sliding-window caches cannot shift them. Returns the same dict as
        intervention/steer.py:generate plus the captured prefill states."""
        from steer import Steer
        torch = self.torch
        steer = steer or Steer(self.model, 1, None)
        states = {}
        handles = self.capture_hooks(capture, states) if capture else []
        t0 = time.perf_counter()
        with torch.inference_mode():
            try:
                steer.start = len(ids) - 1
                out = self._forward(ids, 0, None, True, 1)
            finally:
                for h in handles:
                    h.remove()
            logits, cache, generated, n = out.logits[0, -1], out.past_key_values, [], len(ids)
            for i in range(cap):
                token = int(logits.argmax())
                generated.append(token)
                if token in self.stop or i == cap - 1:
                    break
                steer.start = 0
                out = self._forward([token], n, cache, True, 1)
                logits, cache, n = out.logits[0, -1], out.past_key_values, n + 1
            del cache
        natural = generated[-1] in self.stop
        token_ids = generated[:-1] if natural else generated
        return dict(generated_ids=generated, token_ids=token_ids, truncated=not natural, states=states,
                    termination='cap' if not natural else 'turn_end' if generated[-1] == self.turn_end else 'other_stop',
                    seconds=time.perf_counter() - t0)

    def text(self, token_ids):
        return self.tok.decode(token_ids, skip_special_tokens=True)

    def states(self, ids, layers):
        store = {}
        handles = self.capture_hooks(layers, store)
        try:
            with self.torch.inference_mode():
                self._forward(ids, 0, None, False, 1)
        finally:
            for h in handles:
                h.remove()
        return store

    def tf_logprobs(self, ids, reference, steer):
        """Teacher-forced log-probabilities of each reference position, float32 on the CPU (vocabularies up to 256k)."""
        torch = self.torch
        full = ids + reference[:-1]
        steer.start = len(ids) - 1
        with torch.inference_mode():
            out = self._forward(full, 0, None, False, len(reference))
            logits = out.logits[0]
            assert logits.shape[0] == len(reference)
            chunks = [torch.log_softmax(logits[i:i + 128].float(), dim=-1).cpu() for i in range(0, len(reference), 128)]
        del out, logits
        return torch.cat(chunks)

    @staticmethod
    def kl_summary(ref_logp, logp, reference):
        import torch
        kl = torch.cat([(ref_logp[i:i + 256].exp() * (ref_logp[i:i + 256] - logp[i:i + 256])).sum(-1) for i in range(0, len(reference), 256)]).clamp_min(0)
        target = torch.tensor(reference)
        return dict(kl_mean=float(kl.mean()), kl_first=float(kl[0]), kl_max=float(kl.max()),
                    top1_agreement=float((ref_logp.argmax(-1) == logp.argmax(-1)).float().mean()),
                    ref_token_logprob_mean=float(logp.gather(1, target[:, None]).mean()), positions=len(reference))
