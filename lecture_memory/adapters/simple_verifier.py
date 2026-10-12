
"""
adapters/simple_verifier.py

Implements IEvidenceVerifier.

Checks generated answer and explanation claims against the
retrieved multimodal lecture evidence.

The verifier is intentionally lightweight and conservative.
It does not use an additional language model.

Verification principles:

1. Both ANSWER and EXPLANATION are checked.

2. A factual claim must be directly supported by the
   lecturer speech or explicit visual content.

3. Evidence items are checked independently. Evidence from
   multiple unrelated sections is never blindly combined.

4. Generated topic/concept metadata is NOT treated as factual
   evidence.

5. Inferential statements are rejected unless the complete
   resulting statement is directly present in one evidence item.

6. No domain-specific vocabulary is hardcoded.

7. Unsupported claims are never silently accepted.

The verifier is a grounding safety layer, not a semantic reasoner.
It intentionally prefers rejecting uncertain paraphrases over
accepting unsupported information.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from lecture_memory.domain.models import (
    GeneratedAnswer,
    RetrievedEvidence,
    VerificationResult,
    ClaimVerification,
)
from lecture_memory.domain.enums import VerificationOutcome
from lecture_memory.interfaces.verifier import IEvidenceVerifier
from lecture_memory.observability.logger import get_logger


logger = get_logger(__name__)


class SimpleEvidenceVerifier(IEvidenceVerifier):

    # ----------------------------------------------------------
    # Common words that do not provide useful evidence overlap.
    # ----------------------------------------------------------

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "during",
        "each",
        "for",
        "from",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "more",
        "most",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "under",
        "used",
        "using",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
        "lecture",
        "lecturer",
        "mentioned",
        "mentions",
        "says",
        "said",
        "shows",
        "shown",
        "describes",
        "described",
        "states",
        "state",
        "explains",
        "explained",
    }

    # ----------------------------------------------------------
    # Phrases that introduce an inference or conclusion.
    #
    # These are domain-independent.
    #
    # When one of these appears, the verifier becomes stricter:
    # the COMPLETE claim must be directly supported by a single
    # lecture evidence item.
    # ----------------------------------------------------------

    _INFERENCE_MARKERS = (
        "this implies",
        "this means",
        "this shows",
        "this indicates",
        "therefore",
        "hence",
        "thus",
        "as a result",
        "consequently",
        "which means",
        "which implies",
        "which shows",
        "which indicates",
        "from this we can conclude",
        "we can conclude",
        "it follows that",
        "this demonstrates",
    )

    # ----------------------------------------------------------
    # Main verification
    # ----------------------------------------------------------

    def verify(
        self,
        answer: GeneratedAnswer,
        evidence: List[RetrievedEvidence],
    ) -> VerificationResult:

        # ------------------------------------------------------
        # No evidence
        # ------------------------------------------------------

        if not evidence:
            logger.warning(
                "verification_skipped | reason=no_evidence"
            )

            return VerificationResult(
                answer=answer,
                claim_verifications=[],
                overall_outcome=(
                    VerificationOutcome.INSUFFICIENT_EVIDENCE
                ),
                verified_answer=None,
            )

        # ------------------------------------------------------
        # Extract factual claims from BOTH answer and explanation.
        # ------------------------------------------------------

        claims = self._extract_claims(answer)

        if not claims:
            logger.warning(
                "verification_skipped | "
                "reason=no_verifiable_claims"
            )

            return VerificationResult(
                answer=answer,
                claim_verifications=[],
                overall_outcome=(
                    VerificationOutcome.INSUFFICIENT_EVIDENCE
                ),
                verified_answer=None,
            )

        # ------------------------------------------------------
        # Build individual evidence records.
        #
        # IMPORTANT:
        # Only lecturer speech and explicit visual content are
        # treated as factual evidence.
        #
        # topic/subject/concept labels are generated metadata
        # and therefore intentionally excluded.
        # ------------------------------------------------------

        evidence_records: List[Tuple[str, str]] = []

        for retrieved in evidence:
            concept = retrieved.concept

            speech_text = (
                concept.speech_text.strip()
                if concept.speech_text
                else ""
            )

            visual_text = (
                concept.visual_description.strip()
                if concept.visual_description
                else ""
            )

            evidence_parts = []

            if speech_text:
                evidence_parts.append(speech_text)

            if visual_text:
                evidence_parts.append(visual_text)

            evidence_text = " ".join(
                evidence_parts
            ).strip()

            if not evidence_text:
                continue

            evidence_records.append(
                (
                    concept.concept_id,
                    evidence_text,
                )
            )

        if not evidence_records:
            logger.warning(
                "verification_failed | "
                "reason=evidence_contains_no_text"
            )

            return VerificationResult(
                answer=answer,
                claim_verifications=[
                    ClaimVerification(
                        claim=claim,
                        outcome=(
                            VerificationOutcome.UNSUPPORTED
                        ),
                        supporting_evidence_ids=[],
                        confidence=0.0,
                    )
                    for claim in claims
                ],
                overall_outcome=(
                    VerificationOutcome.UNSUPPORTED
                ),
                verified_answer=None,
            )

        # ------------------------------------------------------
        # Verify every claim independently.
        # ------------------------------------------------------

        verifications: List[ClaimVerification] = []

        for claim in claims:
            verification = self._verify_claim(
                claim,
                evidence_records,
            )

            verifications.append(verification)

            logger.debug(
                "claim_verification | "
                f"claim={claim[:120]} | "
                f"outcome={verification.outcome.value} | "
                f"confidence={verification.confidence:.3f} | "
                f"supporting={len(verification.supporting_evidence_ids)}"
            )

        # ------------------------------------------------------
        # Overall result.
        #
        # The generated answer is accepted only when every
        # extracted claim is supported.
        # ------------------------------------------------------

        supported_count = sum(
            1
            for verification in verifications
            if verification.outcome
            == VerificationOutcome.SUPPORTED
        )

        partial_count = sum(
            1
            for verification in verifications
            if verification.outcome
            == VerificationOutcome.PARTIAL
        )

        unsupported_count = sum(
            1
            for verification in verifications
            if verification.outcome
            == VerificationOutcome.UNSUPPORTED
        )

        if (
            supported_count == len(verifications)
            and len(verifications) > 0
        ):
            overall = VerificationOutcome.SUPPORTED
            verified_answer = answer.answer

        elif unsupported_count > 0:
            # Any unsupported factual claim prevents the
            # generated answer from being accepted.
            overall = VerificationOutcome.UNSUPPORTED
            verified_answer = None

        elif partial_count > 0:
            overall = VerificationOutcome.PARTIAL
            verified_answer = None

        else:
            overall = (
                VerificationOutcome.INSUFFICIENT_EVIDENCE
            )
            verified_answer = None

        logger.info(
            "verification_completed | "
            f"claims={len(verifications)} | "
            f"supported={supported_count} | "
            f"partial={partial_count} | "
            f"unsupported={unsupported_count} | "
            f"outcome={overall.value}"
        )

        return VerificationResult(
            answer=answer,
            claim_verifications=verifications,
            overall_outcome=overall,
            verified_answer=verified_answer,
        )

    # ----------------------------------------------------------
    # Verify one claim
    # ----------------------------------------------------------

    def _verify_claim(
        self,
        claim: str,
        evidence_records: List[Tuple[str, str]],
    ) -> ClaimVerification:

        claim = self._clean_claim(claim)

        if not claim:
            return ClaimVerification(
                claim=claim,
                outcome=VerificationOutcome.UNSUPPORTED,
                supporting_evidence_ids=[],
                confidence=0.0,
            )

        # ------------------------------------------------------
        # Determine whether the claim contains an explicit
        # inferential bridge.
        # ------------------------------------------------------

        contains_inference = self._contains_inference_marker(
            claim
        )

        supporting_ids: List[str] = []
        best_score = 0.0

        # ------------------------------------------------------
        # Check every evidence item independently.
        # ------------------------------------------------------

        for concept_id, evidence_text in evidence_records:

            # --------------------------------------------------
            # INFERENCE CLAIM
            #
            # For an inferential claim, a small matching phrase
            # such as "transport layer" is NOT enough.
            #
            # The complete factual content must be present in
            # the same evidence item.
            # --------------------------------------------------

            if contains_inference:
                if self._complete_claim_supported(
                    claim,
                    evidence_text,
                ):
                    supporting_ids.append(concept_id)
                    best_score = 1.0

                continue

            # --------------------------------------------------
            # NORMAL CLAIM
            # --------------------------------------------------

            score = self._direct_support_score(
                claim,
                evidence_text,
            )

            best_score = max(
                best_score,
                score,
            )

            if self._is_directly_supported(
                claim,
                evidence_text,
                score,
            ):
                supporting_ids.append(concept_id)

        supporting_ids = list(
            dict.fromkeys(supporting_ids)
        )

        # ------------------------------------------------------
        # Inference claims are supported only when the complete
        # claim occurs in one evidence item.
        # ------------------------------------------------------

        if contains_inference:
            if supporting_ids:
                return ClaimVerification(
                    claim=claim,
                    outcome=VerificationOutcome.SUPPORTED,
                    supporting_evidence_ids=supporting_ids,
                    confidence=best_score,
                )

            if best_score > 0.0:
                return ClaimVerification(
                    claim=claim,
                    outcome=VerificationOutcome.PARTIAL,
                    supporting_evidence_ids=[],
                    confidence=best_score,
                )

            return ClaimVerification(
                claim=claim,
                outcome=VerificationOutcome.UNSUPPORTED,
                supporting_evidence_ids=[],
                confidence=0.0,
            )

        # ------------------------------------------------------
        # Normal directly-supported claim.
        # ------------------------------------------------------

        if supporting_ids:
            return ClaimVerification(
                claim=claim,
                outcome=VerificationOutcome.SUPPORTED,
                supporting_evidence_ids=supporting_ids,
                confidence=best_score,
            )

        # ------------------------------------------------------
        # Some overlap exists, but it was not sufficient to
        # establish direct support.
        # ------------------------------------------------------

        if best_score > 0.0:
            return ClaimVerification(
                claim=claim,
                outcome=VerificationOutcome.PARTIAL,
                supporting_evidence_ids=[],
                confidence=best_score,
            )

        return ClaimVerification(
            claim=claim,
            outcome=VerificationOutcome.UNSUPPORTED,
            supporting_evidence_ids=[],
            confidence=0.0,
        )

    # ----------------------------------------------------------
    # Extract claims from answer + explanation
    # ----------------------------------------------------------

    def _extract_claims(
        self,
        answer: GeneratedAnswer,
    ) -> List[str]:

        claims: List[str] = []

        sections = [
            answer.answer,
            answer.explanation,
        ]

        for section in sections:
            for claim in self._split_into_claims(section):
                self._add_unique_claim(
                    claims,
                    claim,
                )

        return claims

    # ----------------------------------------------------------
    # Split generated text into candidate claims
    # ----------------------------------------------------------

    def _split_into_claims(
        self,
        text: str,
    ) -> List[str]:

        if not text:
            return []

        cleaned = text.strip()

        # Remove common generated section headers.
        cleaned = re.sub(
            r"(?im)^\s*\*{1,3}\s*"
            r"(ANSWER|EXPLANATION)"
            r"\s*:\s*\*{1,3}\s*$",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"(?im)^\s*"
            r"(ANSWER|EXPLANATION)"
            r"\s*:\s*$",
            "",
            cleaned,
        )

        # Remove Markdown emphasis without changing words.
        cleaned = re.sub(
            r"[\*\_\`]+",
            "",
            cleaned,
        )

        # Split on sentence-ending punctuation.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            cleaned,
        )

        claims: List[str] = []

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Ignore fragments that do not contain enough
            # meaningful words to represent a factual claim.
            if len(
                self._normalise_tokens(sentence)
            ) < 3:
                continue

            claims.append(sentence)

        return claims

    # ----------------------------------------------------------
    # Duplicate claim detection
    # ----------------------------------------------------------

    def _is_duplicate_claim(
        self,
        claim: str,
        existing_claims: List[str],
    ) -> bool:

        claim_tokens = set(
            self._normalise_tokens(claim)
        )

        if not claim_tokens:
            return True

        for existing in existing_claims:

            existing_tokens = set(
                self._normalise_tokens(existing)
            )

            if not existing_tokens:
                continue

            intersection = (
                claim_tokens.intersection(
                    existing_tokens
                )
            )

            smaller_set_size = min(
                len(claim_tokens),
                len(existing_tokens),
            )

            if (
                smaller_set_size > 0
                and len(intersection)
                == smaller_set_size
            ):
                return True

        return False

    # ----------------------------------------------------------
    # Add unique claim
    # ----------------------------------------------------------

    def _add_unique_claim(
        self,
        claims: List[str],
        claim: str,
    ) -> None:

        if self._is_duplicate_claim(
            claim,
            claims,
        ):
            return

        claims.append(claim)

    # ----------------------------------------------------------
    # Determine whether a normal claim is directly supported.
    #
    # IMPORTANT:
    #
    # We intentionally removed the previous "meaningful phrase"
    # rule and arbitrary score threshold.
    #
    # A short matching phrase must never be enough to validate
    # an entire generated claim.
    # ----------------------------------------------------------

    def _is_directly_supported(
        self,
        claim: str,
        evidence_text: str,
        score: float,
    ) -> bool:

        claim_tokens = self._normalise_tokens(
            claim
        )

        evidence_tokens = self._normalise_tokens(
            evidence_text
        )

        if not claim_tokens or not evidence_tokens:
            return False

        # ------------------------------------------------------
        # Safest lexical case:
        #
        # Every meaningful word in the claim occurs in the
        # SAME evidence item.
        #
        # Evidence from different items is never combined.
        # ------------------------------------------------------

        claim_set = set(claim_tokens)
        evidence_set = set(evidence_tokens)

        if claim_set.issubset(evidence_set):
            return True

        # ------------------------------------------------------
        # Exact normalized claim occurrence.
        #
        # This handles cases where punctuation differs but the
        # actual statement appears directly in the evidence.
        # ------------------------------------------------------

        normalized_claim = " ".join(
            claim_tokens
        )

        normalized_evidence = " ".join(
            evidence_tokens
        )

        if (
            normalized_claim
            and normalized_claim in normalized_evidence
        ):
            return True

        return False

    # ----------------------------------------------------------
    # Complete support for an inferential claim.
    #
    # This is deliberately stricter than normal claims.
    #
    # Example:
    #
    # Claim:
    # "TCP ... implies that the transport layer provides
    # end-to-end communication."
    #
    # Evidence:
    # "TCP is the foundation for pretty much everything..."
    #
    # The claim contains additional factual content that is not
    # present in the evidence, so it is rejected.
    # ----------------------------------------------------------

    def _complete_claim_supported(
        self,
        claim: str,
        evidence_text: str,
    ) -> bool:

        claim_tokens = self._normalise_tokens(
            claim
        )

        evidence_tokens = self._normalise_tokens(
            evidence_text
        )

        if not claim_tokens or not evidence_tokens:
            return False

        claim_set = set(claim_tokens)
        evidence_set = set(evidence_tokens)

        # The complete meaningful claim must be represented
        # within the same evidence item.
        if claim_set.issubset(evidence_set):
            return True

        normalized_claim = " ".join(
            claim_tokens
        )

        normalized_evidence = " ".join(
            evidence_tokens
        )

        return bool(
            normalized_claim
            and normalized_claim in normalized_evidence
        )

    # ----------------------------------------------------------
    # Direct support score
    #
    # This score is used only for diagnostic confidence and
    # partial-support reporting.
    #
    # It is NOT used as an arbitrary acceptance threshold.
    # ----------------------------------------------------------

    def _direct_support_score(
        self,
        claim: str,
        evidence_text: str,
    ) -> float:

        claim_tokens = self._normalise_tokens(
            claim
        )

        evidence_tokens = self._normalise_tokens(
            evidence_text
        )

        if not claim_tokens or not evidence_tokens:
            return 0.0

        claim_set = set(claim_tokens)
        evidence_set = set(evidence_tokens)

        overlap = claim_set.intersection(
            evidence_set
        )

        coverage = (
            len(overlap) / len(claim_set)
        )

        return coverage

    # ----------------------------------------------------------
    # Inference marker detection
    # ----------------------------------------------------------

    def _contains_inference_marker(
        self,
        claim: str,
    ) -> bool:

        lowered = re.sub(
            r"\s+",
            " ",
            claim.lower(),
        ).strip()

        return any(
            marker in lowered
            for marker in self._INFERENCE_MARKERS
        )

    # ----------------------------------------------------------
    # Claim cleanup
    # ----------------------------------------------------------

    def _clean_claim(
        self,
        claim: str,
    ) -> str:

        cleaned = claim.strip()

        cleaned = re.sub(
            r"^\s*(?:[-•]\s*)+",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"[\*\_\`]+",
            "",
            cleaned,
        )

        return cleaned.strip()

    # ----------------------------------------------------------
    # Token normalization
    # ----------------------------------------------------------

    def _normalise_tokens(
        self,
        text: str,
    ) -> List[str]:

        if not text:
            return []

        text = text.lower()

        tokens = re.findall(
            r"[a-z0-9]+",
            text,
        )

        useful_tokens: List[str] = []

        for token in tokens:

            if len(token) <= 2:
                continue

            if token in self._STOPWORDS:
                continue

            useful_tokens.append(token)

        return useful_tokens
