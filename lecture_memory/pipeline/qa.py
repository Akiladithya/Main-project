
"""
pipeline/qa.py

Responsibility:

Answer a student question using retrieved multimodal lecture evidence.

Pipeline:

Question
    ↓
Multimodal retrieval
    ↓
Temporal evidence expansion
    ↓
Qwen answer generation
    ↓
TEMPORARILY RETURN GENERATED ANSWER
    ↓
Evidence verification will be re-enabled later

IMPORTANT:
Verification is temporarily bypassed so that we can test
retrieval + Qwen answer generation independently.

The verifier code itself is NOT removed.
"""

from __future__ import annotations

from lecture_memory.domain.models import (
    GeneratedAnswer,
    VerificationResult,
)
from lecture_memory.domain.enums import VerificationOutcome
from lecture_memory.interfaces.retriever import IRetriever
from lecture_memory.interfaces.generator import IAnswerGenerator
from lecture_memory.interfaces.verifier import IEvidenceVerifier
from lecture_memory.observability.logger import (
    get_logger,
    stage_timer,
    log_retrieval,
    log_verification,
)


logger = get_logger(__name__)


class QuestionAnsweringPipeline:

    def __init__(
        self,
        retriever: IRetriever,
        generator: IAnswerGenerator,
        verifier: IEvidenceVerifier,
        max_verification_retries: int = 1,
    ) -> None:

        self._retriever = retriever
        self._generator = generator
        self._verifier = verifier
        self._max_retries = max_verification_retries

    def answer(
        self,
        question: str,
        video_id: str | None = None,
    ) -> VerificationResult:

        with stage_timer(
            logger,
            "qa_pipeline",
            question_len=len(question),
        ):

            # ==================================================
            # 1. INITIAL MULTIMODAL RETRIEVAL
            # ==================================================

            evidence = self._retriever.retrieve(
                question,
                video_id=video_id,
                top_k=10,
                expand_context_sec=30.0,
            )

            log_retrieval(
                logger,
                question,
                len(evidence),
                evidence[0].combined_score
                if evidence
                else None,
            )

            # ==================================================
            # No evidence
            # ==================================================

            if not evidence:

                logger.warning(
                    "no_evidence_found | returning insufficient result"
                )

                return self._build_insufficient_result(
                    question=question,
                    video_id=video_id,
                    evidence_ids=[],
                    reason=(
                        "No relevant lecture content "
                        "was found for this question."
                    ),
                )

            # ==================================================
            # 2. GENERATE INITIAL ANSWER
            # ==================================================

            logger.info(
                "qa_generation_started | "
                f"evidence_count={len(evidence)}"
            )

            answer = self._generator.generate(
                question,
                evidence,
            )

            # ==================================================
            # TEMPORARY VERIFICATION BYPASS
            # ==================================================
            #
            # For now, we are testing only:
            #
            # Retrieval → Qwen → Answer
            #
            # The verifier is intentionally bypassed.
            # It will be re-enabled later after we finish
            # testing answer generation.
            # ==================================================

            logger.info(
                "qa_verification_bypassed | "
                "reason=testing_answer_generation"
            )

            return VerificationResult(
                answer=answer,
                claim_verifications=[],
                overall_outcome=VerificationOutcome.SUPPORTED,
                verified_answer=answer.answer,
            )

            # ==================================================
            # VERIFICATION CODE BELOW
            # ==================================================
            #
            # This code is intentionally kept in the file so
            # that we can re-enable verification later.
            #
            # It is currently unreachable because of the
            # temporary return above.
            # ==================================================

            # ==================================================
            # 3. VERIFY INITIAL ANSWER
            # ==================================================

            result = self._verifier.verify(
                answer,
                evidence,
            )

            self._log_verification_result(result)

            # ==================================================
            # 4. Decide whether broader retrieval is required
            # ==================================================

            needs_retry = self._needs_retry(result)

            if (
                needs_retry
                and self._max_retries > 0
            ):

                logger.info(
                    "verification_retry | "
                    "reason=insufficient_support | "
                    "expanding_retrieval"
                )

                # ==================================================
                # 5. BROADER MULTIMODAL RETRIEVAL
                # ==================================================

                expanded = self._retriever.retrieve(
                    question,
                    video_id=video_id,
                    top_k=20,
                    expand_context_sec=60.0,
                )

                log_retrieval(
                    logger,
                    question,
                    len(expanded),
                    expanded[0].combined_score
                    if expanded
                    else None,
                )

                # ==================================================
                # 6. REGENERATE USING BROADER CONTEXT
                # ==================================================

                if expanded:

                    logger.info(
                        "qa_retry_generation_started | "
                        f"evidence_count={len(expanded)}"
                    )

                    answer = self._generator.generate(
                        question,
                        expanded,
                    )

                    # ==================================================
                    # 7. VERIFY RETRIED ANSWER
                    # ==================================================

                    result = self._verifier.verify(
                        answer,
                        expanded,
                    )

                    self._log_verification_result(
                        result,
                        retry=True,
                    )

                    # ==================================================
                    # 8. CRITICAL SAFETY CHECK
                    # ==================================================

                    if self._needs_retry(result):

                        logger.warning(
                            "qa_answer_rejected | "
                            "reason=insufficient_evidence_after_retry"
                        )

                        return self._build_insufficient_result(
                            question=question,
                            video_id=video_id,
                            evidence_ids=[
                                item.concept.concept_id
                                for item in expanded
                            ],
                            reason=(
                                "The retrieved lecture evidence does not "
                                "provide enough support for a reliable "
                                "answer to this question."
                            ),
                            claim_verifications=(
                                result.claim_verifications
                            ),
                        )

                else:

                    logger.warning(
                        "verification_retry_failed | "
                        "no_expanded_evidence"
                    )

                    return self._build_insufficient_result(
                        question=question,
                        video_id=video_id,
                        evidence_ids=[
                            item.concept.concept_id
                            for item in evidence
                        ],
                        reason=(
                            "The available lecture evidence was not "
                            "sufficient to verify the generated answer."
                        ),
                        claim_verifications=(
                            result.claim_verifications
                        ),
                    )

            # ==================================================
            # 9. FINAL SAFETY CHECK
            # ==================================================

            if self._needs_retry(result):

                logger.warning(
                    "qa_answer_rejected | "
                    "reason=final_verification_failed"
                )

                return self._build_insufficient_result(
                    question=question,
                    video_id=video_id,
                    evidence_ids=[
                        item.concept.concept_id
                        for item in evidence
                    ],
                    reason=(
                        "The available lecture evidence does not "
                        "sufficiently support a reliable answer."
                    ),
                    claim_verifications=(
                        result.claim_verifications
                    ),
                )

            # ==================================================
            # 10. RETURN VERIFIED RESULT
            # ==================================================

            logger.info(
                "qa_pipeline_completed | "
                f"outcome={result.overall_outcome.value}"
            )

            return result

    # ==========================================================
    # Determine whether generated answer needs another attempt
    # ==========================================================

    def _needs_retry(
        self,
        result: VerificationResult,
    ) -> bool:

        if (
            result.overall_outcome
            == VerificationOutcome.UNSUPPORTED
        ):
            return True

        if (
            result.overall_outcome
            == VerificationOutcome.PARTIAL
        ):
            return True

        if (
            result.overall_outcome
            == VerificationOutcome.INSUFFICIENT_EVIDENCE
        ):
            return True

        return False

    # ==========================================================
    # Build safe insufficient-evidence result
    # ==========================================================

    def _build_insufficient_result(
        self,
        question: str,
        video_id: str | None,
        evidence_ids: list[str],
        reason: str,
        claim_verifications=None,
    ) -> VerificationResult:

        safe_answer = GeneratedAnswer(
            question=question,
            answer=(
                "I couldn't find enough information in the "
                "uploaded lecture to answer this question reliably."
            ),
            explanation=reason,
            timestamp_start=None,
            timestamp_end=None,
            video_id=video_id,
            evidence_ids=evidence_ids,
        )

        return VerificationResult(
            answer=safe_answer,
            claim_verifications=(
                claim_verifications
                if claim_verifications is not None
                else []
            ),
            overall_outcome=(
                VerificationOutcome.INSUFFICIENT_EVIDENCE
            ),
            verified_answer=None,
        )

    # ==========================================================
    # Verification logging
    # ==========================================================

    def _log_verification_result(
        self,
        result: VerificationResult,
        retry: bool = False,
    ) -> None:

        n_supported = sum(
            1
            for verification in result.claim_verifications
            if verification.outcome
            == VerificationOutcome.SUPPORTED
        )

        n_partial = sum(
            1
            for verification in result.claim_verifications
            if verification.outcome
            == VerificationOutcome.PARTIAL
        )

        n_unsupported = sum(
            1
            for verification in result.claim_verifications
            if verification.outcome
            == VerificationOutcome.UNSUPPORTED
        )

        logger.info(
            "verification_result | "
            f"retry={retry} | "
            f"claims={len(result.claim_verifications)} | "
            f"supported={n_supported} | "
            f"partial={n_partial} | "
            f"unsupported={n_unsupported} | "
            f"overall={result.overall_outcome.value}"
        )

        log_verification(
            logger,
            len(result.claim_verifications),
            n_supported,
            n_unsupported,
        )

