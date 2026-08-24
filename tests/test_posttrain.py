"""Post-training tests: SFT masking / packing, generation, and GRPO.

Properties checked:

* the SFT loss is computed on assistant tokens only: it equals a hand-computed
  cross-entropy over assistant positions, and zeroing the mask drops it to zero;
* packed conversations do not attend across their boundaries -- a token's logits
  are invariant to edits in another segment, but not once blocking is off;
* generation is deterministic under a fixed seed and terminates on ``<|eos|>``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from config import Config, swiglu_hidden  # noqa: E402
from mini_gpt.chat_template import build_prompt  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.generate import generate, generate_reply  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402
from mini_gpt.posttrain.sft import (  # noqa: E402
    PAD_SEGMENT,
    pack_conversations,
    sft_loss,
    synthetic_sft_conversations,
    train_sft,
)
from mini_gpt.tokenizer import MiniTokenizer  # noqa: E402


def _tiny_cfg(**overrides) -> Config:
    base = dict(
        name="tiny",
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=16,
        mlp_hidden=swiglu_hidden(64),
        context=48,
        window=16,
        micro_batch=4,
        grad_accum=2,
        warmup_steps=5,
        max_steps=100,
        use_muon=False,
        compile=False,
        dtype="float32",
        use_triton=False,
    )
    base.update(overrides)
    return Config(**base)


@pytest.fixture(scope="module")
def tokenizer() -> MiniTokenizer:
    convs = synthetic_sft_conversations(400, seed=0)
    text = [m["content"] for c in convs for m in c]
    return MiniTokenizer.train(text, vocab_size=512, min_frequency=1)


# ==========================================================================
# packing, masking, and cross-conversation isolation
# ==========================================================================

def test_pack_shapes_and_padding(tokenizer):
    convs = synthetic_sft_conversations(20, seed=1)
    packed = pack_conversations(convs, tokenizer, seq_len=48)
    assert packed.input_ids.shape == packed.targets.shape
    assert packed.input_ids.shape[1] == 48
    assert packed.loss_mask.shape == packed.input_ids.shape
    assert packed.segment_ids.shape == packed.input_ids.shape
    # Padding positions carry the pad segment and are excluded from the loss.
    pad = packed.segment_ids == PAD_SEGMENT
    assert torch.all(packed.loss_mask[pad] == 0)


def test_sft_loss_is_assistant_only(tokenizer):
    convs = synthetic_sft_conversations(24, seed=2)
    packed = pack_conversations(convs, tokenizer, seq_len=48)
    seed_everything(0)
    model = MiniGPT(_tiny_cfg())

    ids, tgt, mask, seg = next(packed.microbatches(4))

    # (a) hand-computed CE over assistant positions equals the model's loss.
    logits, model_loss = model(ids, tgt, loss_mask=mask)  # no segment blocking here
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_tgt = tgt.reshape(-1)
    flat_mask = mask.reshape(-1).bool()
    manual = F.cross_entropy(flat_logits[flat_mask], flat_tgt[flat_mask])
    assert flat_mask.any(), "fixture produced no assistant tokens"
    assert torch.allclose(model_loss, manual, atol=1e-5)

    # (b) zeroing the mask (removing the supervised tokens) drops the loss to 0.
    model.fused_loss = True  # chunked CE returns 0 (not NaN) when nothing is supervised
    zero_mask = torch.zeros_like(mask)
    _, loss_zero = model(ids, tgt, loss_mask=zero_mask)
    assert float(loss_zero.detach()) == pytest.approx(0.0, abs=1e-6)


def test_packed_conversations_do_not_attend_across_boundaries(tokenizer):
    # Two whole conversations packed into one row; edits in segment 0 must not
    # change segment 1's logits when the segment mask is on.
    convs = synthetic_sft_conversations(2, seed=3)
    packed = pack_conversations(convs, tokenizer, seq_len=64)
    ids = packed.input_ids[:1]
    seg = packed.segment_ids[:1]
    assert (seg == 0).any() and (seg == 1).any(), "need two segments in one row"

    seed_everything(0)
    model = MiniGPT(_tiny_cfg(context=64)).eval()

    seg0_pos = (seg[0] == 0).nonzero().flatten()
    seg1_pos = (seg[0] == 1).nonzero().flatten()

    ids_edit = ids.clone()
    # Flip a token inside segment 0 to a different id.
    p = int(seg0_pos[len(seg0_pos) // 2])
    ids_edit[0, p] = (int(ids_edit[0, p]) + 1) % model.cfg.vocab_size

    with torch.no_grad():
        base = model(ids, segment_ids=seg)[0][0, seg1_pos]
        edited = model(ids_edit, segment_ids=seg)[0][0, seg1_pos]
        edited_noblock = model(ids_edit, segment_ids=None)[0][0, seg1_pos]

    assert torch.allclose(base, edited, atol=1e-6), "segment 1 leaked info from segment 0"
    # Sanity: without blocking, the later segment *does* see the edit.
    assert not torch.allclose(base, edited_noblock, atol=1e-6)


def test_sft_mask_loss(tokenizer):
    # Combines the two checks above.
    test_sft_loss_is_assistant_only(tokenizer)
    test_packed_conversations_do_not_attend_across_boundaries(tokenizer)


def test_sft_trains_on_synthetic(tokenizer):
    convs = synthetic_sft_conversations(64, seed=4)
    packed = pack_conversations(convs, tokenizer, seq_len=48)
    seed_everything(0)
    cfg = _tiny_cfg(lr_adamw=1e-3)
    model = MiniGPT(cfg)
    losses = train_sft(model, packed, cfg, device="cpu", steps=80)
    assert losses[-1] < losses[0], f"SFT did not train: {losses[0]:.3f} -> {losses[-1]:.3f}"


# ==========================================================================
# generation
# ==========================================================================

class _EosModel(nn.Module):
    """A stub that always predicts ``eos``, to exercise eos-stopping."""

    def __init__(self, vocab: int, eos: int):
        super().__init__()
        self.vocab = vocab
        self.eos = eos
        self._p = nn.Parameter(torch.zeros(1))  # so .parameters()/.to() work

    def forward(self, idx, targets=None, **kw):
        b, t = idx.shape
        logits = torch.zeros(b, t, self.vocab, device=idx.device)
        logits[..., self.eos] = 100.0
        return logits, None


def test_generate_greedy_is_deterministic(tokenizer):
    seed_everything(0)
    model = MiniGPT(_tiny_cfg())
    idx = torch.tensor([build_prompt([{"role": "user", "content": "hi"}], tokenizer)])
    a = generate(model, idx, max_new_tokens=10, temperature=0.0)
    b = generate(model, idx, max_new_tokens=10, temperature=0.0)
    assert torch.equal(a, b)
    assert a.shape[1] == idx.shape[1] + 10  # no eos_id passed -> runs full length


def test_generate_temperature_seed_is_reproducible(tokenizer):
    seed_everything(0)
    model = MiniGPT(_tiny_cfg())
    idx = torch.tensor([build_prompt([{"role": "user", "content": "hi"}], tokenizer)])
    a = generate(model, idx, max_new_tokens=12, temperature=0.8, top_k=20, seed=123)
    b = generate(model, idx, max_new_tokens=12, temperature=0.8, top_k=20, seed=123)
    assert torch.equal(a, b)


def test_generate_stops_on_eos(tokenizer):
    model = _EosModel(tokenizer.vocab_size, tokenizer.eos_id)
    idx = torch.tensor([[tokenizer.bos_id, 5, 6]])
    out = generate(model, idx, max_new_tokens=20, temperature=0.0, eos_id=tokenizer.eos_id)
    # First generated token is eos -> loop breaks immediately after one step.
    assert out.shape[1] == idx.shape[1] + 1
    assert int(out[0, -1]) == tokenizer.eos_id


def test_generate_format_reply_is_wellformed(tokenizer):
    # generate_reply returns only the (eos-terminated) assistant text, decoded.
    model = _EosModel(tokenizer.vocab_size, tokenizer.eos_id)
    reply = generate_reply(model, tokenizer, [{"role": "user", "content": "2+2?"}], max_new_tokens=8)
    assert isinstance(reply, str)
    assert reply == ""  # the stub emits eos immediately -> empty assistant turn


def test_generate_format(tokenizer):
    # Deterministic and eos-terminating.
    test_generate_greedy_is_deterministic(tokenizer)
    test_generate_stops_on_eos(tokenizer)


# ==========================================================================
# GRPO
# ==========================================================================

from mini_gpt.posttrain import grpo  # noqa: E402
from mini_gpt.posttrain.rewards import (  # noqa: E402
    arithmetic_reward,
    countdown_reward,
    extract_final_int,
    gsm8k_reward,
)


# ---------------------------------------------------------------------- rewards

def test_rewards():
    MAX = 16
    # Correct arithmetic scores high; the correctness term is present.
    good = arithmetic_reward("the answer is 8", target=8, terminated=True, n_new_tokens=4, max_new_tokens=MAX)
    assert good.correct == 1.0 and good.total > 1.0

    # Parseable-but-wrong: no correctness, but format shaping gives partial credit.
    wrong = arithmetic_reward("7", target=8, terminated=True, n_new_tokens=1, max_new_tokens=MAX)
    assert wrong.correct == 0.0 and 0.0 < wrong.total < good.total

    # Malformed / non-terminating: no signal at all.
    bad = arithmetic_reward("", target=8, terminated=False, n_new_tokens=0, max_new_tokens=MAX)
    assert bad.total == 0.0
    # Format shaping is strictly informative before correctness is achievable.
    assert wrong.total > bad.total

    # Countdown: correct expression using the operands scores; wrong / illegal don't.
    ok = countdown_reward("3 * 5 = 15", numbers=[3, 5], target=15, terminated=True, n_new_tokens=5, max_new_tokens=MAX)
    assert ok.correct == 1.0
    miss = countdown_reward("3 + 5", numbers=[3, 5], target=15, terminated=True, n_new_tokens=3, max_new_tokens=MAX)
    assert miss.correct == 0.0
    illegal = countdown_reward("3 * 9", numbers=[3, 5], target=27, terminated=True, n_new_tokens=3, max_new_tokens=MAX)
    assert illegal.correct == 0.0  # 9 is not an allowed operand

    # GSM8K: the #### delimiter answer is extracted. Correctness is near-zero at
    # mini, but the scorer itself is exact.
    assert extract_final_int("reasoning ... #### 42") == 42
    g = gsm8k_reward("#### 42", gold=42, terminated=True, n_new_tokens=2, max_new_tokens=MAX)
    assert g.correct == 1.0


# --------------------------------------------------------------- group sampling

def test_group_sampling(tokenizer):
    seed_everything(0)
    model = MiniGPT(_tiny_cfg())
    prompts = [[{"role": "user", "content": "What is 2 + 3?"}]]
    G = 6

    a = grpo.sample_groups(model, tokenizer, prompts, group_size=G, max_new_tokens=8, temperature=1.0, seed=0)
    b = grpo.sample_groups(model, tokenizer, prompts, group_size=G, max_new_tokens=8, temperature=1.0, seed=0)

    assert len(a) == G  # one prompt -> G completions
    assert [c.tokens for c in a] == [c.tokens for c in b]  # seeded + reproducible
    assert len({tuple(c.tokens) for c in a}) > 1  # the group is not degenerate


# ------------------------------------------------------------ advantage + update

def test_group_advantages_and_zero_variance():
    groups = torch.zeros(4, dtype=torch.long)
    adv = grpo.group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]), groups)
    assert adv[0] > 0 and torch.all(adv[1:] < 0)  # best above baseline, rest below
    assert float(adv.sum()) == pytest.approx(0.0, abs=1e-5)  # mean-centred

    # Zero-variance group -> zero advantage (the degenerate case).
    flat = grpo.group_advantages(torch.tensor([0.5, 0.5, 0.5]), torch.zeros(3, dtype=torch.long))
    assert torch.all(flat == 0.0)


def _hand_group(tokenizer, model):
    prompt = build_prompt([{"role": "user", "content": "q"}], tokenizer)
    # Two completions sharing the prompt, differing in their generated token.
    c_hi = grpo.Completion(0, len(prompt), prompt + [10, tokenizer.eos_id], "hi", True, 2)
    c_lo = grpo.Completion(0, len(prompt), prompt + [11, tokenizer.eos_id], "lo", True, 2)
    seqs, mask, groups = grpo.collate([c_hi, c_lo], tokenizer.pad_id)
    return seqs, mask, groups


def test_grpo_update_moves_logprobs_in_the_right_direction(tokenizer):
    seed_everything(0)
    model = MiniGPT(_tiny_cfg())
    seqs, mask, groups = _hand_group(tokenizer, model)
    advantages = torch.tensor([1.0, -1.0])  # first completion good, second bad

    def total_logp():
        lp, m = grpo.token_logprobs(model, seqs, mask)
        per = (lp * m.float()).sum(dim=1)
        return per.detach().clone()

    before = total_logp()
    opt = torch.optim.SGD(model.parameters(), lr=0.02)  # small step, no weight decay -> clean sign
    grpo.grpo_step(model, opt, seqs, mask, advantages, grad_clip=0.0)
    after = total_logp()

    assert after[0] > before[0], "positive-advantage completion should get more likely"
    assert after[1] < before[1], "negative-advantage completion should get less likely"


def test_grpo_zero_variance_group_produces_no_gradient(tokenizer):
    seed_everything(0)
    model = MiniGPT(_tiny_cfg())
    seqs, mask, groups = _hand_group(tokenizer, model)
    advantages = torch.zeros(2)  # equal rewards -> zero advantage

    with torch.no_grad():
        old_logp, _ = grpo.token_logprobs(model, seqs, mask)
    loss = grpo.grpo_loss(model, seqs, mask, advantages, old_logp.detach())
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-7)
    assert grad_norm == pytest.approx(0.0, abs=1e-7)


def test_grpo_reward_rises_on_countdown_format():
    # With a learnable shaped reward (terminate the completion), GRPO drives the
    # mean reward up over steps. This exercises the format signal the reward
    # shaping relies on to avoid stalling at zero.
    convs = synthetic_sft_conversations(200, seed=0)
    text = [m["content"] for c in convs for m in c]
    tok = MiniTokenizer.train(text, vocab_size=512, min_frequency=1)

    seed_everything(0)
    model = MiniGPT(_tiny_cfg())

    def termination_reward(c: grpo.Completion) -> float:
        # Pure format shaping: reward terminating (emitting eos) within budget.
        return 1.0 if c.terminated else 0.0

    import random

    rng = random.Random(0)
    bank = [
        ([{"role": "user", "content": f"What is {rng.randint(0, 9)} + {rng.randint(0, 9)}?"}], termination_reward)
        for _ in range(16)
    ]

    rewards = grpo.run_grpo(
        model, tok, bank, steps=40, group_size=8, prompts_per_step=4,
        max_new_tokens=8, temperature=1.0, top_k=None, lr=1e-2, seed=0,
    )
    first = sum(rewards[:8]) / 8
    last = sum(rewards[-8:]) / 8
    assert last > first, f"GRPO reward did not rise: {first:.3f} -> {last:.3f}"


def test_group_sampling_headline(tokenizer):
    test_group_sampling(tokenizer)


def test_grpo_update(tokenizer):
    # Advantage signs correct, and a zero-variance group is a no-op.
    test_group_advantages_and_zero_variance()
    test_grpo_update_moves_logprobs_in_the_right_direction(tokenizer)
    test_grpo_zero_variance_group_produces_no_gradient(tokenizer)
