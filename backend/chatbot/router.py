# backend/chatbot/router.py
#
# FIXES:
# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — _MIN_SCORE lowered to 2 (from 3).
#   With only 2 stress questions (max score=6) a threshold of 3 meant stress
#   needed 50% hit rate. 2 is already a "several days" answer — clinically
#   meaningful. Raising it too high causes "neutral" for genuinely symptomatic
#   users.
#
# FIX 2 — Normalised scoring to handle unequal question counts.
#   anxiety=3 qs (max 9), depression=3 qs (max 9), stress=2 qs (max 6).
#   Without normalisation stress is always disadvantaged. We compare
#   average-per-question instead of raw sum.
#   avg = raw_sum / question_count
#
# FIX 3 — Priority: anxiety > stress > depression.
#   Depression wins ties too often because its symptoms overlap with anxiety
#   and fatigue. Anxiety first is the safer clinical entry point.
# ─────────────────────────────────────────────────────────────────────────────

# Question counts per domain (must match _SCREENING_DOMAIN_MAP in engine)
_QUESTION_COUNTS = {
    "anxiety":    3,   # feeling_nervous, uncontrollable_worry, restlessness
    "depression": 3,   # feeling_down, loss_of_interest, fatigue
    "stress":     2,   # overwhelmed, irritability
}

# Minimum AVERAGE score (0-3 scale) needed to route to a condition.
# 0.8 = roughly "several days on at least one question" — meaningful signal.
_MIN_AVG = 0.8

# Tie-break priority
_PRIORITY = ["anxiety", "stress", "depression"]


def route_condition(screening_scores: dict) -> str:
    """
    Routes to the condition with the highest normalised screening score.

    Rules:
      - Raw score divided by question count → average score per question.
      - Averages below _MIN_AVG are ignored.
      - If no condition meets the threshold → "neutral".
      - Ties broken by: anxiety > stress > depression.

    Returns one of: "anxiety" | "stress" | "depression" | "neutral"
    """
    averages = {}
    for condition in ("anxiety", "stress", "depression"):
        raw = screening_scores.get(condition, 0)
        count = _QUESTION_COUNTS.get(condition, 1)
        avg = raw / count
        if avg >= _MIN_AVG:
            averages[condition] = avg

    print(
        f"[router] raw={screening_scores}  "
        f"averages={averages}  "
        f"min_avg={_MIN_AVG}",
        flush=True,
    )

    if not averages:
        return "neutral"

    top_avg = max(averages.values())

    # Allow a small tolerance for "tie" — within 0.1 of each other
    tied = [k for k, v in averages.items() if abs(v - top_avg) < 0.1]

    if len(tied) == 1:
        return tied[0]

    for condition in _PRIORITY:
        if condition in tied:
            return condition

    return tied[0]