# nlu.py
#
# Natural Language Understanding module for SentiCare.
#
# POSITION IN PIPELINE:
#   STT → NLU → EmotionAnalyzer
#
# WHAT IT DOES:
#   1. detectIntent()     — distress / denial / help_seeking / positive_engagement / neutral
#   2. analyzeSentiment() — positive / negative / neutral polarity
#   3. extractKeywords()  — clinical keywords signalling anxiety, stress, sadness
#
# FIX HISTORY:
#   FIX 1 — negation checked BEFORE counting positive words ("not good" → negative)
#   FIX 2 — positive bank uses first-person anchors ("Are you good?" no longer scores)
#   FIX 3 — negated wellbeing ("I am not good") → intent=distress, sentiment=negative
#   FIX 4 — windowed anchor-based positive matching; "I am REALLY happy" now works;
#            added positive_engagement intent for clearly happy speech
#   FIX 5 — curly-apostrophe normalisation (\u2019 → ') so Whisper STT output like
#            "I\u2019m" is treated the same as "I'm" for anchor and negation matching
#
# NO NEW DEPENDENCIES — stdlib + regex only.

import re

# ── Unicode normalisation helper ──────────────────────────────────────────────
# Whisper STT often outputs curly/smart quotes. Normalise before any matching.

def _normalise(text: str) -> str:
    return (
        text
        .replace("\u2019", "'")   # right single quotation mark  →  '
        .replace("\u2018", "'")   # left  single quotation mark  →  '
        .replace("\u201C", '"')   # left  double quotation mark  →  "
        .replace("\u201D", '"')   # right double quotation mark  →  "
        .replace("\u2014", " ")   # em-dash                      →  space
        .replace("\u2013", " ")   # en-dash                      →  space
    )

# ── Clinical keyword banks ────────────────────────────────────────────────────

_ANXIETY_KEYWORDS_EN = [
    "panic", "panic attack", "anxiety", "anxious", "nervous", "worried",
    "worry", "fear", "scared", "terrified", "racing heart", "palpitation",
    "chest tight", "can't breathe", "shortness of breath", "dizzy",
    "trembling", "shaking", "sweating", "overwhelmed", "doom", "dread",
    "phobia", "restless", "on edge", "keyed up", "tense",
]

_STRESS_KEYWORDS_EN = [
    "stress", "stressed", "pressure", "deadline", "overload", "burnout",
    "exhausted", "drained", "frustrated", "irritable", "snapping",
    "workload", "too much", "can't cope", "can't manage", "falling behind",
    "academic pressure", "exam", "assignment", "no time", "overwhelmed",
    "headache", "sleep problems", "insomnia", "can't sleep",
]

_SADNESS_KEYWORDS_EN = [
    "sad", "cry", "crying", "tears", "hopeless", "helpless", "empty",
    "numb", "worthless", "lonely", "alone", "depressed", "depression",
    "miserable", "grief", "loss", "heartbroken", "low mood", "dark",
]

_HELP_KEYWORDS_EN = [
    "help", "need help", "support", "talk to someone", "counselor",
    "therapist", "doctor", "professional", "hotline", "crisis",
    "don't know what to do", "please help",
]

_ANXIETY_KEYWORDS_UR = [
    "گھبراہٹ", "پریشانی", "بے چینی", "خوف", "ڈر", "دھڑکن", "کانپنا",
    "سانس", "دباؤ محسوس", "گھبرا", "نروس",
]
_STRESS_KEYWORDS_UR = [
    "دباؤ", "ذہنی دباؤ", "تھکاوٹ", "پریشان", "نیند نہیں", "سر درد",
    "چڑچڑاپن", "غصہ", "امتحان", "کام کا بوجھ", "وقت نہیں",
]
_SADNESS_KEYWORDS_UR = [
    "اداسی", "اداس", "رونا", "مایوسی", "ناامید", "تنہا", "اکیلا",
    "بے بس", "بے کار", "خالی پن", "ڈپریشن",
]
_HELP_KEYWORDS_UR = [
    "مدد", "مدد چاہیے", "کسی سے بات", "ڈاکٹر", "ماہر", "ہیلپ لائن",
]

# ── Wellbeing keyword bank (FIX 3) ───────────────────────────────────────────
# Negated form ("I am not good/fine/okay") → distress signal.

_WELLBEING_KEYWORDS_EN = [
    "good", "fine", "okay", "ok", "well", "alright",
    "great", "happy", "calm", "relaxed", "better",
]
_WELLBEING_KEYWORDS_UR = [
    "بہتر", "ٹھیک", "اچھا", "خوش", "سکون", "پرسکون",
]

# ── Positive keyword bank (FIX 4) ────────────────────────────────────────────
# Bare words — runtime anchor check ensures only first-person uses count.

_POSITIVE_KEYWORDS_EN = frozenset({
    "happy", "happiness", "glad", "pleased", "content", "fine",
    "okay", "good", "great", "better", "well", "calm", "relaxed",
    "relieved", "grateful", "thankful", "hopeful", "alright",
    "excited", "exciting", "excitement", "thrilled", "elated",
    "joyful", "joy", "wonderful", "fantastic", "amazing", "excellent",
    "awesome", "cheerful", "enthusiastic", "energized", "motivated",
    "inspired", "optimistic", "proud", "confident",
    "improving", "recovering", "progress",
    "nice", "love", "loving", "enjoy", "enjoying",
})

_POSITIVE_KEYWORDS_UR = frozenset({
    "خوش", "بہتر", "اچھا", "ٹھیک", "سکون", "پرسکون",
    "شکرگزار", "امیدوار", "پرجوش", "اچھا محسوس",
})

# First-person anchors — positive keyword must appear near one of these.
_FP_ANCHORS_EN = frozenset({
    "i", "i'm", "im", "i've", "ive", "i'd", "id",
    "we", "we're", "we've", "my", "me", "myself", "our",
})
_FP_ANCHORS_UR = frozenset({"میں", "ہم", "مجھے", "ہمیں", "میرا", "ہمارا"})

_ANCHOR_WINDOW = 6   # tokens left/right to search for anchor

# ── Negation patterns ─────────────────────────────────────────────────────────

_NEGATION_EN = re.compile(
    r"\b(not|no|never|don't|dont|didn't|didnt|isn't|isnt|"
    r"wasn't|wasnt|won't|wont|can't|cant|couldn't|couldnt|"
    r"hardly|barely|neither|nor|am not|i'm not|i am not)\b",
    re.IGNORECASE,
)
_NEGATION_UR = re.compile(
    r"(نہیں|نہ|کبھی نہیں|مت|بالکل نہیں)"
)

# ── Intent constants ──────────────────────────────────────────────────────────

INTENT_DISTRESS  = "distress"
INTENT_DENIAL    = "denial"
INTENT_HELP_SEEK = "help_seeking"
INTENT_POSITIVE  = "positive_engagement"
INTENT_NEUTRAL   = "neutral"


# ─────────────────────────────────────────────────────────────────────────────

class NLU:
    """
    Natural Language Understanding for SentiCare.
    STT → NLU → EmotionAnalyzer

    Public result keys:
        intent              str   — distress | denial | help_seeking |
                                    positive_engagement | neutral
        sentiment           str   — positive | negative | neutral
        sentiment_score     float — -1.0 … +1.0
        keywords            dict  — {anxiety, stress, sadness, help}
        positive_keywords   list  — matched positive keyword strings
        language            str   — "en" | "ur"
        negation_found      bool  — negation near clinical keyword
        negated_wellbeing   bool  — negation near wellbeing word (FIX 3)
        anxiety_boost       float — 0.0–0.3 additive boost for EmotionAnalyzer
        stress_boost        float
        sadness_boost       float
    """

    def __init__(self):
        self.intent:            str   = INTENT_NEUTRAL
        self.sentiment:         str   = "neutral"
        self.sentiment_score:   float = 0.0
        self.keywords:          dict  = {"anxiety": [], "stress": [], "sadness": [], "help": []}
        self.positive_keywords: list  = []
        self.language:          str   = "en"
        self.negation_found:    bool  = False
        self.negated_wellbeing: bool  = False

    # ── Public entry point ────────────────────────────────────────────────────

    def analyze(self, text: str, language: str = "en") -> dict:
        """
        Full NLU pipeline.

        Parameters
        ----------
        text     : raw STT transcript (may be empty)
        language : "en" or "ur"

        Returns
        -------
        dict — all NLU fields + boost values for EmotionAnalyzer
        """
        self.language = language

        # FIX 5 — normalise curly quotes from Whisper before anything else
        text = _normalise((text or "").strip())

        if not text:
            return self._empty_result()

        self.keywords          = self._extract_keywords(text, language)
        self.negation_found, \
        self.negated_wellbeing = self._detect_negation(text, language)
        self.positive_keywords = self._extract_positive_keywords(text, language)
        self.sentiment, \
        self.sentiment_score   = self._score_sentiment()
        self.intent            = self._classify_intent()
        boosts                 = self._compute_boosts()

        result = {
            "intent":            self.intent,
            "sentiment":         self.sentiment,
            "sentiment_score":   round(self.sentiment_score, 3),
            "keywords":          self.keywords,
            "positive_keywords": self.positive_keywords,
            "language":          self.language,
            "negation_found":    self.negation_found,
            "negated_wellbeing": self.negated_wellbeing,
            "anxiety_boost":     boosts["anxiety"],
            "stress_boost":      boosts["stress"],
            "sadness_boost":     boosts["sadness"],
        }

        print(
            f"[NLU] intent={result['intent']}  sentiment={result['sentiment']}  "
            f"score={result['sentiment_score']}  negation={result['negation_found']}  "
            f"negated_wellbeing={result['negated_wellbeing']}  "
            f"pos_kw={result['positive_keywords']}  "
            f"anxiety_boost={result['anxiety_boost']}  "
            f"stress_boost={result['stress_boost']}  "
            f"sadness_boost={result['sadness_boost']}",
            flush=True,
        )
        return result

    # ── Public aliases (match class diagram) ──────────────────────────────────

    def detectIntent(self, text: str, language: str = "en") -> str:
        self.analyze(text, language)
        return self.intent

    def analyzeSentiment(self, text: str, language: str = "en") -> dict:
        self.analyze(text, language)
        return {"sentiment": self.sentiment, "score": self.sentiment_score}

    def extractKeywords(self, text: str, language: str = "en") -> dict:
        self.language = language
        return self._extract_keywords(_normalise(text), language)

    # ── Private: keyword extraction ───────────────────────────────────────────

    def _extract_keywords(self, text: str, language: str) -> dict:
        text_lower = text.lower()

        if language == "ur":
            banks = {
                "anxiety": _ANXIETY_KEYWORDS_UR,
                "stress":  _STRESS_KEYWORDS_UR,
                "sadness": _SADNESS_KEYWORDS_UR,
                "help":    _HELP_KEYWORDS_UR,
            }
        else:
            banks = {
                "anxiety": _ANXIETY_KEYWORDS_EN,
                "stress":  _STRESS_KEYWORDS_EN,
                "sadness": _SADNESS_KEYWORDS_EN,
                "help":    _HELP_KEYWORDS_EN,
            }

        found = {cat: [kw for kw in bank if kw in text_lower]
                 for cat, bank in banks.items()}

        if any(found.values()):
            print(f"[NLU] Clinical keywords: {found}", flush=True)

        return found

    # ── Private: positive keyword extraction (FIX 4 + FIX 5) ─────────────────

    def _extract_positive_keywords(self, text: str, language: str) -> list:
        """
        Windowed anchor-based matching.

        For each token that matches the positive bank:
          1. Check ±ANCHOR_WINDOW tokens for a first-person anchor.
             → no anchor  means the sentence is directed at someone else ("Are you happy?")
          2. Check the same window for negation.
             → negation present  means "I am NOT happy" → skip
          3. Both checks pass → valid positive hit.

        FIX 5: text is already normalised by analyze() so "I\u2019m" → "I'm"
        before tokenisation, ensuring "i'm" appears in _FP_ANCHORS_EN.
        """
        if language == "ur":
            pos_bank    = _POSITIVE_KEYWORDS_UR
            anchors     = _FP_ANCHORS_UR
            neg_pattern = _NEGATION_UR
        else:
            pos_bank    = _POSITIVE_KEYWORDS_EN
            anchors     = _FP_ANCHORS_EN
            neg_pattern = _NEGATION_EN

        tokens  = text.lower().split()
        matched = []

        for i, token in enumerate(tokens):
            clean = re.sub(r"[^\w']", "", token)
            if clean not in pos_bank:
                continue

            lo  = max(0, i - _ANCHOR_WINDOW)
            hi  = min(len(tokens), i + _ANCHOR_WINDOW + 1)
            win_tokens = tokens[lo:hi]
            win_str    = " ".join(win_tokens)

            # Must have a first-person anchor in window
            if not any(re.sub(r"[^\w']", "", t) in anchors for t in win_tokens):
                print(
                    f"[NLU] '{clean}' skipped — no first-person anchor "
                    f"in window '{win_str}'",
                    flush=True,
                )
                continue

            # Must NOT be negated
            if neg_pattern.search(win_str):
                print(
                    f"[NLU] '{clean}' skipped — negated in window '{win_str}'",
                    flush=True,
                )
                continue

            print(f"[NLU] Positive keyword accepted: '{clean}'", flush=True)
            matched.append(clean)

        return matched

    # ── Private: negation detection (FIX 3 + FIX 5) ──────────────────────────

    def _detect_negation(self, text: str, language: str) -> tuple:
        """
        Returns (negation_found, negated_wellbeing).

        negation_found    — negation appeared near a clinical keyword
        negated_wellbeing — negation appeared near a wellbeing word
                            ("I am not good/fine/okay")

        FIX 5: text is pre-normalised so "I\u2019m not" → "I'm not",
        which matches the compiled _NEGATION_EN pattern.
        """
        pattern = _NEGATION_UR if language == "ur" else _NEGATION_EN
        tokens  = text.lower().split()

        clinical_all = (
            _ANXIETY_KEYWORDS_EN + _STRESS_KEYWORDS_EN + _SADNESS_KEYWORDS_EN +
            _ANXIETY_KEYWORDS_UR + _STRESS_KEYWORDS_UR + _SADNESS_KEYWORDS_UR
        )
        wellbeing = (
            _WELLBEING_KEYWORDS_UR if language == "ur" else _WELLBEING_KEYWORDS_EN
        )

        negation_found    = False
        negated_wellbeing = False

        for i, token in enumerate(tokens):
            if not pattern.search(token):
                continue

            window = " ".join(tokens[i: i + 5])

            if not negation_found:
                for kw in clinical_all:
                    if kw in window:
                        print(
                            f"[NLU] Negation near clinical keyword '{kw}' "
                            f"in window: '{window}'",
                            flush=True,
                        )
                        negation_found = True
                        break

            if not negated_wellbeing:
                for kw in wellbeing:
                    if kw in window:
                        print(
                            f"[NLU] Negated wellbeing word '{kw}' "
                            f"in window: '{window}'",
                            flush=True,
                        )
                        negated_wellbeing = True
                        break

            if negation_found and negated_wellbeing:
                break

        return negation_found, negated_wellbeing

    # ── Private: sentiment scoring ────────────────────────────────────────────

    def _score_sentiment(self) -> tuple:
        """
        positive_count = validated positive keywords (FIX 4)
        negative_count = clinical count + negated_wellbeing flag (FIX 3)
        score = (pos - neg) / max(total, 1), clamped [-1, +1]
        """
        negative_count = (
            len(self.keywords.get("anxiety", [])) +
            len(self.keywords.get("stress",  [])) +
            len(self.keywords.get("sadness", []))
        )
        if self.negated_wellbeing:
            negative_count += 1

        positive_count = len(self.positive_keywords)
        total          = negative_count + positive_count

        if total == 0:
            return "neutral", 0.0

        score = max(-1.0, min(1.0, (positive_count - negative_count) / total))

        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return label, score

    # ── Private: intent classification ───────────────────────────────────────

    def _classify_intent(self) -> str:
        """
        Priority:
          1. help keywords          → help_seeking
          2. negation + clinical    → denial
          3. clinical distress      → distress
          4. negated wellbeing      → distress
          5. positive keywords      → positive_engagement
          6. fallback               → neutral
        """
        has_clinical = (
            len(self.keywords.get("anxiety", [])) > 0 or
            len(self.keywords.get("stress",  [])) > 0 or
            len(self.keywords.get("sadness", [])) > 0
        )
        has_help = len(self.keywords.get("help", [])) > 0

        if has_help:
            return INTENT_HELP_SEEK
        if self.negation_found and has_clinical:
            return INTENT_DENIAL
        if has_clinical:
            return INTENT_DISTRESS
        if self.negated_wellbeing:
            return INTENT_DISTRESS
        if self.positive_keywords:
            return INTENT_POSITIVE
        return INTENT_NEUTRAL

    # ── Private: EmotionAnalyzer boost values ─────────────────────────────────

    def _compute_boosts(self) -> dict:
        """
        Each clinical keyword → +0.08, capped at 0.30 per category.
        Halved when negation_found (user is denying the feeling).
        negated_wellbeing with no clinical keywords → small baseline boost.
        """
        def _boost(count: int) -> float:
            raw = min(count * 0.08, 0.30)
            return round(raw * 0.5, 3) if self.negation_found else round(raw, 3)

        boosts = {
            "anxiety": _boost(len(self.keywords.get("anxiety", []))),
            "stress":  _boost(len(self.keywords.get("stress",  []))),
            "sadness": _boost(len(self.keywords.get("sadness", []))),
        }

        if self.negated_wellbeing and not any([
            self.keywords.get("anxiety"),
            self.keywords.get("stress"),
            self.keywords.get("sadness"),
        ]):
            boosts["stress"]  = max(boosts["stress"],  0.08)
            boosts["sadness"] = max(boosts["sadness"], 0.08)

        return boosts

    # ── Private: empty result ─────────────────────────────────────────────────

    def _empty_result(self) -> dict:
        return {
            "intent":            INTENT_NEUTRAL,
            "sentiment":         "neutral",
            "sentiment_score":   0.0,
            "keywords":          {"anxiety": [], "stress": [], "sadness": [], "help": []},
            "positive_keywords": [],
            "language":          self.language,
            "negation_found":    False,
            "negated_wellbeing": False,
            "anxiety_boost":     0.0,
            "stress_boost":      0.0,
            "sadness_boost":     0.0,
        }


# ── Smoke tests ───────────────────────────────────────────────────────────────
# python nlu.py

if __name__ == "__main__":
    nlu = NLU()

    tests = [
        # (desc, text, lang, exp_intent, exp_sentiment)
        (
            "Bug 1 — 'I am not good'",
            "Hello chatbot, how are you? I am not good. Are you good? Are you listening?",
            "en", "distress", "negative",
        ),
        (
            "Bug 2 — Whisper curly quote: I\u2019m really happy (FIX 5 target)",
            "Happy Chinese! Hello! I\u2019m really happy today that you\u2019re listening "
            "me from Sargodha University and I\u2019m doing my final year project and "
            "we\u2019ll submit this on Monday.",
            "en", "positive_engagement", "positive",
        ),
        (
            "Bug 2 straight-quote variant",
            "I am really happy today because we are moving towards the next step of "
            "making a good chat board and this is really nice seeing you completing "
            "and doing a lot of effort.",
            "en", "positive_engagement", "positive",
        ),
        (
            "Bot-directed — should NOT be positive",
            "Are you happy? Are you good? How are you doing?",
            "en", "neutral", "neutral",
        ),
        (
            "Explicit denial of clinical keyword",
            "I do not feel anxious at all.",
            "en", "denial", "neutral",
        ),
        (
            "Clear clinical distress",
            "I am having a panic attack and I can't breathe.",
            "en", "distress", "negative",
        ),
        (
            "Help seeking",
            "I need help, can I talk to a counselor?",
            "en", "help_seeking", "neutral",
        ),
        (
            "Genuine positive with adverb",
            "I'm feeling so much better today, I feel really calm and relaxed.",
            "en", "positive_engagement", "positive",
        ),
        (
            "Negated positive",
            "I am not happy at all, everything feels wrong.",
            "en", "distress", "negative",
        ),
        (
            "Excited about progress",
            "I am so excited about this, we are really motivated and inspired.",
            "en", "positive_engagement", "positive",
        ),
        (
            "Urdu distress",
            "مجھے بہت پریشانی ہے اور نیند نہیں آتی",
            "ur", "distress", "negative",
        ),
    ]

    print("=" * 72)
    passed = 0
    for desc, text, lang, exp_intent, exp_sentiment in tests:
        print(f"\nTEST : {desc}")
        print(f"TEXT : {text[:90]}{'…' if len(text) > 90 else ''}")
        r = nlu.analyze(text, lang)
        ok = r["intent"] == exp_intent and r["sentiment"] == exp_sentiment
        if ok:
            passed += 1
        status = "PASS ✓" if ok else "FAIL ✗"
        print(
            f"[{status}]  intent={r['intent']} (exp={exp_intent})  "
            f"sentiment={r['sentiment']} (exp={exp_sentiment})  "
            f"score={r['sentiment_score']}  pos_kw={r['positive_keywords'][:3]}"
        )
        print("-" * 72)

    print(f"\n{'='*72}")
    print(f"Results: {passed}/{len(tests)} passed")