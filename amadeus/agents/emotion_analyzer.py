"""
Fast CPU-based emotion analyzer for Kurisu's English responses.
Uses VADER sentiment analysis with Kurisu-specific enhancements.
"""

from __future__ import annotations

from typing import Dict

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None


class KurisuEmotionAnalyzer:
    """
    Lightweight emotion analyzer optimized for CPU performance.

    Analyzes English text to determine Kurisu's emotional state and maps
    it to VRM facial expressions. Uses VADER for base sentiment analysis
    with Kurisu-specific keyword boosters for character-accurate emotions.

    Performance: <1ms per analysis on CPU
    Memory: ~30MB
    """

    def __init__(self):
        """Initialize VADER analyzer and Kurisu-specific keyword sets."""
        self.vader = (
            SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        )

        # Kurisu personality-specific keywords (tsundere!)
        self.tsundere_keywords = [
            "baka",
            "idiot",
            "hmph",
            "fool",
            "dummy",
            "stupid",
            "whatever",
            "don't get me wrong",
            "not like i",
            "it's not like",
        ]

        # Excited/happy keywords (scientific enthusiasm)
        self.excited_keywords = [
            "fascinating",
            "incredible",
            "amazing",
            "brilliant",
            "excellent",
            "wonderful",
            "remarkable",
            "extraordinary",
            "magnificent",
            "phenomenal",
            "success",
            "eureka",
        ]

        # Sad/concerned keywords
        self.sad_keywords = [
            "unfortunately",
            "sadly",
            "sorry",
            "regret",
            "worry",
            "concerned",
            "unfortunate",
            "disappointed",
            "terrible",
            "awful",
            "bad news",
        ]

        # Surprised keywords
        self.surprised_keywords = [
            "what",
            "really",
            "impossible",
            "unexpected",
            "can't believe",
            "shocking",
            "unbelievable",
            "astonishing",
            "wait",
            "huh",
        ]

        # Relaxed/thoughtful keywords (analytical mode)
        self.thoughtful_keywords = [
            "consider",
            "analyze",
            "theory",
            "hypothesis",
            "perhaps",
            "possibly",
            "might",
            "could be",
            "let me think",
            "interesting",
            "hmm",
        ]

        # Map emotions to VRM expression names
        # Based on extracted VRM expressions: happy, angry, sad, relaxed, surprised, neutral
        self.vrm_expression_map = {
            "happy": "happy",
            "angry": "angry",
            "sad": "sad",
            "surprised": "surprised",
            "relaxed": "relaxed",
            "neutral": "neutral",
        }

    def analyze(self, text: str) -> Dict:
        """
        Analyze emotion in English text and map to VRM expression.

        Performance: <1ms on CPU (very fast!)

        Args:
            text: English text from Kurisu's response

        Returns:
            Dictionary containing:
            - emotion: Detected emotion name
            - vrm_expression: VRM expression name to trigger
            - intensity: Expression intensity (0.0-1.0)
            - confidence: Confidence in detection (0.0-1.0)
            - vader_scores: Raw VADER sentiment scores
        """
        if not text or not text.strip():
            return self._neutral_response()

        text_lower = text.lower()

        # Priority 1: Check Kurisu-specific tsundere patterns (highest priority!)
        if any(kw in text_lower for kw in self.tsundere_keywords):
            return {
                "emotion": "angry",
                "vrm_expression": "angry",
                "intensity": 0.6,  # Moderate intensity (tsundere is subtle)
                "confidence": 0.95,
                "detected_pattern": "tsundere",
            }

        # Get VADER sentiment scores
        vader_scores = (
            self.vader.polarity_scores(text)
            if self.vader
            else {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        )
        compound = vader_scores["compound"]

        # Priority 2: Check for strong emotional keywords with context

        # Surprised (requires punctuation for emphasis)
        if any(kw in text_lower for kw in self.surprised_keywords):
            has_emphasis = "!" in text or "?" in text
            if has_emphasis:
                return {
                    "emotion": "surprised",
                    "vrm_expression": "surprised",
                    "intensity": 0.85,
                    "confidence": 0.90,
                    "vader_scores": vader_scores,
                    "detected_pattern": "surprised_keywords",
                }

        # Excited/Happy (scientific enthusiasm + positive sentiment)
        excited_match = any(kw in text_lower for kw in self.excited_keywords)
        if excited_match or compound > 0.5:
            intensity = min(1.0, max(0.6, compound + 0.3))
            return {
                "emotion": "happy",
                "vrm_expression": "happy",
                "intensity": intensity,
                "confidence": 0.85,
                "vader_scores": vader_scores,
                "detected_pattern": (
                    "excited" if excited_match else "positive_sentiment"
                ),
            }

        # Sad/Concerned
        sad_match = any(kw in text_lower for kw in self.sad_keywords)
        if sad_match or compound < -0.3:
            intensity = min(0.9, max(0.5, abs(compound)))
            return {
                "emotion": "sad",
                "vrm_expression": "sad",
                "intensity": intensity,
                "confidence": 0.80,
                "vader_scores": vader_scores,
                "detected_pattern": "sad" if sad_match else "negative_sentiment",
            }

        # Angry (negative sentiment without sadness markers)
        if compound < -0.1 and not sad_match:
            intensity = min(0.8, abs(compound))
            return {
                "emotion": "angry",
                "vrm_expression": "angry",
                "intensity": intensity,
                "confidence": 0.75,
                "vader_scores": vader_scores,
                "detected_pattern": "negative_sentiment",
            }

        # Thoughtful/Relaxed (analytical mode)
        thoughtful_match = any(kw in text_lower for kw in self.thoughtful_keywords)
        if thoughtful_match:
            return {
                "emotion": "relaxed",
                "vrm_expression": "relaxed",
                "intensity": 0.5,
                "confidence": 0.80,
                "vader_scores": vader_scores,
                "detected_pattern": "thoughtful",
            }

        # Mildly positive
        if compound > 0.1:
            return {
                "emotion": "happy",
                "vrm_expression": "happy",
                "intensity": min(0.6, compound),
                "confidence": 0.70,
                "vader_scores": vader_scores,
                "detected_pattern": "mild_positive",
            }

        # Default to relaxed for neutral scientific responses
        return {
            "emotion": "relaxed",
            "vrm_expression": "relaxed",
            "intensity": 0.4,
            "confidence": 0.65,
            "vader_scores": vader_scores,
            "detected_pattern": "neutral",
        }

    def _neutral_response(self) -> Dict:
        """Return neutral emotion response for empty/invalid input."""
        return {
            "emotion": "neutral",
            "vrm_expression": "neutral",
            "intensity": 0.3,
            "confidence": 1.0,
            "detected_pattern": "empty_input",
        }

    def get_available_expressions(self) -> list[str]:
        """Return list of available VRM expressions."""
        return list(self.vrm_expression_map.values())
