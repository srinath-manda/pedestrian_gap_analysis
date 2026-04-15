"""attribute_classifier.py — DeepFace-based gender and age group classifier."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

VALID_GENDERS = {"Man", "Woman", "Male", "Female"}


@dataclass
class PedestrianAttributes:
    gender: str    # "Male" | "Female" | "Unknown"
    age_group: str  # "Young" | "Middle" | "Old" | "Unknown"


class AttributeClassifier:
    """
    Classifies gender and age group from a cropped pedestrian bounding-box image
    using DeepFace.  All exceptions are caught; "Unknown" is returned on failure.
    """

    def classify(self, crop: np.ndarray) -> PedestrianAttributes:
        """
        Analyse a BGR crop with DeepFace.
        Returns PedestrianAttributes with gender and age_group.
        """
        try:
            from deepface import DeepFace  # lazy import — heavy library

            results = DeepFace.analyze(
                crop,
                actions=["gender", "age"],
                enforce_detection=False,
                silent=True,
            )
            # DeepFace may return a list or a single dict
            result = results[0] if isinstance(results, list) else results

            raw_gender = result.get("dominant_gender", "Unknown")
            gender = self._normalise_gender(raw_gender)

            age_val = result.get("age", None)
            age_group = self.age_to_group(int(age_val)) if age_val is not None else "Unknown"

        except Exception as exc:  # noqa: BLE001
            logger.warning("[AttributeClassifier] DeepFace failed: %s", exc)
            gender = "Unknown"
            age_group = "Unknown"

        return PedestrianAttributes(gender=gender, age_group=age_group)

    # ── Static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalise_gender(raw: str) -> str:
        """Map DeepFace gender strings to Male/Female/Unknown."""
        mapping = {
            "man": "Male",
            "male": "Male",
            "woman": "Female",
            "female": "Female",
        }
        return mapping.get(raw.lower(), "Unknown")

    @staticmethod
    def age_to_group(age: int) -> str:
        """
        Bucket an integer age into Young / Middle / Old.
        Young  : age ≤ 29
        Middle : 30 ≤ age ≤ 59
        Old    : age ≥ 60
        """
        if age <= 29:
            return "Young"
        if age <= 59:
            return "Middle"
        return "Old"
