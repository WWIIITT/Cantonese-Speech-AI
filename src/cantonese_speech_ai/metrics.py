from collections.abc import Sequence


def _distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def error_rate(ref: Sequence[str], hyp: Sequence[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _distance(ref, hyp) / len(ref)


def char_error_rate(reference: str, hypothesis: str) -> float:
    return error_rate(list(reference), list(hypothesis))


def word_error_rate(reference: str, hypothesis: str) -> float:
    return error_rate(reference.split(), hypothesis.split())
