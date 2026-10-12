"""
adapters/qwen_generator.py

Qwen2.5-VL based answer generator.

The model generates a study-oriented answer using ONLY the
retrieved multimodal lecture evidence.

The project is intended for Computer Science lectures, including
technical subjects such as Computer Networks. However, this
adapter remains domain-independent and does not hardcode specific
concepts, terminology, timestamps, or question types.

Grounding priority:

1. Lecturer speech
2. Explicit visual content
3. Speech + visual combination
4. Generated metadata only for locating context

The generator is deliberately conservative:

- It must extract supported facts before composing the answer.
- It must never complete missing facts from pretrained knowledge.
- Unsupported details must be omitted rather than guessed.
"""

from __future__ import annotations

from typing import List
import re

import torch

from lecture_memory.adapters.qwen_runtime import QwenRuntime
from lecture_memory.domain.models import (
    RetrievedEvidence,
    GeneratedAnswer,
)
from lecture_memory.interfaces.generator import IAnswerGenerator
from lecture_memory.observability.logger import get_logger


logger = get_logger(__name__)


# ==========================================================
# QWEN GROUNDING PROMPT
# ==========================================================

_PROMPT = """
You are a STRICT lecture-grounded study assistant for
Computer Science lectures.

You answer a student's question about an uploaded lecture video.

The lecture may contain technical concepts, definitions,
explanations, examples, diagrams, terminology, procedures,
comparisons, or other educational content.

Your answer must be based ONLY on the supplied lecture evidence.

==========================================================
CLOSED-BOOK RULE
==========================================================

The supplied LECTURE EVIDENCE is your ONLY source of factual
information.

You have pretrained knowledge, but you MUST NOT use it.

Do NOT use:

- world knowledge
- textbook knowledge
- memory
- assumptions
- likely facts
- common knowledge
- domain knowledge
- information about the topic that is not explicitly present
  in the supplied lecture evidence

Think of the lecture evidence as a CLOSED BOOK.

If a fact is missing from the evidence, OMIT the fact.

Do NOT complete incomplete information using your own knowledge.

==========================================================
STRICT TECHNICAL GROUNDING
==========================================================

A technical term, protocol, algorithm, architecture, component,
method, data structure, programming concept, networking concept,
or other named concept does NOT give you permission to explain
what it normally means outside the lecture.

For every technical concept:

- Use only the properties explicitly stated in the lecture.
- Do not provide its standard textbook definition unless the
  lecture itself provides that definition.
- Do not list standard properties that the lecture does not state.
- Do not add standard applications or use cases.
- Do not add standard advantages or disadvantages.
- Do not add standard mechanisms or implementation details.
- Do not add related concepts merely because they are normally
  associated with the concept.
- Do not expand an abbreviation using pretrained knowledge unless
  the expansion appears in the lecture evidence.
- Do not explain a named protocol, algorithm, method, or component
  using information that is absent from the lecture.

IMPORTANT:

Mentioning a concept is NOT evidence for all facts normally
associated with that concept.

For example, if the lecture only states:

"The transport layer provides end-to-end communication."

You may state that the lecture describes the transport layer as
providing end-to-end communication.

You may NOT automatically add other standard properties,
mechanisms, protocols, reliability features, or implementation
details unless they are explicitly present in the evidence.

==========================================================
NO INFERENTIAL BRIDGES
==========================================================

Do NOT create a new factual claim by reasoning from one lecture
statement to another.

Do NOT use phrases such as:

- "this implies"
- "this means"
- "therefore"
- "hence"
- "thus"
- "which means"
- "which implies"
- "as a result"
- "consequently"
- "this shows that"
- "this indicates that"
- "from this we can conclude"
- "this demonstrates that"

when those phrases introduce information that the lecturer did
not explicitly state.

A fact being logically likely, technically true, or normally
associated with another fact does NOT make it lecture-supported.

For example:

Evidence:

"The transport layer provides end-to-end communication."

Evidence:

"TCP is the foundation for pretty much everything we've looked
at so far."

You may state these two statements separately if relevant.

You may NOT combine them into:

"The transport layer includes TCP and therefore provides reliable
end-to-end communication."

That sentence introduces relationships and properties that were
not explicitly stated in the supplied evidence.

When a connection between two facts is not explicitly stated,
keep the facts separate or omit the connection.

==========================================================
QUESTION
==========================================================

{question}

==========================================================
LECTURE EVIDENCE
==========================================================

{evidence_block}

==========================================================
CRITICAL GROUNDING PROCEDURE
==========================================================

Before writing the final answer, internally perform these steps:

STEP 1 — EXTRACT

Identify only statements that are explicitly supported by:

A. LECTURER SPEECH

B. VISUAL CONTENT

C. A direct combination of A and B describing the same
   lecture section

Do not use outside knowledge.

STEP 2 — CHECK

For every factual statement you intend to write, ask:

"Where exactly is this supported in the supplied lecture
evidence?"

If you cannot point to the supplied speech or visual content,
DO NOT write the statement.

STEP 3 — CHECK FOR INFERENCE

Ask:

"Did I derive this statement from another statement instead of
finding it explicitly in the lecture evidence?"

If yes, DELETE the derived statement.

Do not use logical reasoning, textbook knowledge, or technical
knowledge to fill the gap.

STEP 4 — REMOVE OUTSIDE KNOWLEDGE

After checking each statement, remove anything that comes from
your pretrained knowledge.

In particular, remove facts that were added because they are:

- normally associated with the concept
- standard textbook information
- technically true in general
- expected in a Computer Science explanation
- useful background knowledge
- likely to help answer the question

STEP 5 — COMPOSE

Write the answer only from the remaining supported facts.

Never add a fact merely because it is:

- technically common
- scientifically plausible
- normally true
- commonly known
- logically likely
- associated with a named concept
- expected for the subject
- found in standard textbooks

==========================================================
EVIDENCE PRIORITY
==========================================================

Use evidence in this order:

1. Explicit lecturer speech

2. Explicit visual content

3. Speech + visual content describing the same lecture section

4. Generated concept/topic labels only for context

Generated labels are NOT factual evidence.

If a generated label conflicts with speech or visual content,
IGNORE the generated label.

==========================================================
LECTURER SPEECH
==========================================================

LECTURER SPEECH is the strongest textual source.

You may state something as something the lecturer said ONLY
when that information actually appears in LECTURER SPEECH.

You may paraphrase the lecturer's statement when the meaning
remains faithful to the supplied speech.

Do not add missing details.

==========================================================
VISUAL CONTENT
==========================================================

VISUAL CONTENT contains observations extracted from lecture
frames.

Use only information explicitly described there.

You may report:

- visible names
- visible labels
- visible fields
- visible interface elements
- visible diagrams
- visible objects
- visible text
- visible relationships that are explicitly described

Do NOT infer information that is not explicitly described.

For example:

If the visual description says that a diagram contains two
labeled components connected by an arrow, you may describe those
visible components and the stated connection.

You may NOT infer the purpose, behavior, advantage, or technical
meaning of those components unless the lecture explicitly
provides that information.

==========================================================
MULTIMODAL CONNECTION
==========================================================

When speech and visual content refer to the SAME lecture section,
you may connect them.

For example:

If the lecturer explains a technical concept and the visual
content explicitly describes a diagram illustrating that same
concept, you may explain that the diagram corresponds to the
lecture explanation.

Do NOT create a connection solely because timestamps are close.

Do NOT create a technical relationship merely because two
concepts occur in the same lecture section.

==========================================================
NO ENTITY KNOWLEDGE
==========================================================

A named entity or technical term does NOT give you permission
to use your pretrained knowledge about it.

For example, if the lecture mentions a technical protocol,
algorithm, class, database concept, operating-system concept,
or networking concept, you may report only what the supplied
lecture evidence explicitly states about it.

Do NOT add:

- definitions not stated in the lecture
- properties not stated in the lecture
- advantages
- disadvantages
- applications
- examples
- classifications
- historical information
- technical behavior
- numerical values

unless those details are explicitly present in the supplied
lecture evidence.

==========================================================
NO INFERENCE
==========================================================

Do NOT infer:

- causes
- effects
- purposes
- benefits
- disadvantages
- classifications
- relationships
- technical properties
- scientific properties
- numerical values
- reasons
- motivations
- consequences

unless explicitly supported.

Even if the inference appears obvious, do not make it.

==========================================================
PARTIAL FACTS
==========================================================

If the lecture provides only part of a fact, report only the
supported part.

Example:

Evidence:

"The protocol uses a sequence number."

Allowed:

"The lecture states that the protocol uses a sequence number."

Not allowed:

"The sequence number prevents duplicate packets."

The second statement adds information that was not explicitly
provided.

==========================================================
FACTUAL CONSERVATISM
==========================================================

When deciding between:

A. a detailed answer that contains some unsupported information

and

B. a smaller answer containing only explicitly supported facts

ALWAYS choose B.

It is better to omit information than to guess.

==========================================================
QUESTION RELEVANCE
==========================================================

Answer the actual question.

Use the supplied evidence that is relevant to the question.

Do not include unrelated facts simply because they appear in
the evidence.

For different question types, answer using only the supported
information appropriate to the question.

For example:

- For a definition question, provide the supported definition.
- For a "why" question, provide a reason only if the lecture
  explicitly gives one.
- For a comparison question, compare only properties explicitly
  stated for the compared concepts.
- For a process question, describe only the steps explicitly
  presented in the lecture.
- For an example question, use only examples actually present
  in the lecture.
- For a broad question, include multiple directly relevant
  supported facts when available.

Do NOT use these question types as permission to supply
information from pretrained knowledge.

==========================================================
STUDY-QUALITY ANSWER
==========================================================

When sufficient evidence exists, provide a clear,
study-oriented answer.

Prefer:

- simple technical language
- direct explanations
- logically organized points when useful
- terminology used by the lecturer
- concise explanations supported by the lecture

For a broad question, naturally organize the answer around
the relevant information explicitly presented in the lecture.

Do not add external examples.

Do not expand the answer using your own knowledge.

==========================================================
INSUFFICIENT INFORMATION
==========================================================

If the supplied evidence does not contain enough information to
answer the question reliably, say:

"The lecture does not provide enough information to answer this question."

Do not guess.

==========================================================
NO INTERNAL INFORMATION
==========================================================

Never mention:

- retrieval
- embeddings
- vector databases
- RAG
- Qwen
- models
- prompts
- evidence IDs
- UUIDs
- system architecture
- internal concepts
- retrieval scores
- screenshots
- pipeline stages

==========================================================
OUTPUT FORMAT
==========================================================

Return ONLY:

ANSWER:

<answer based strictly on supported lecture facts>

EXPLANATION:

<short explanation based strictly on the supplied speech and
visual content>

Do not add anything before ANSWER:

Do not add anything after the EXPLANATION section.

==========================================================
FINAL SELF-CHECK
==========================================================

Before producing the output, silently check every factual claim.

For EVERY factual claim:

1. Locate its support in LECTURER SPEECH or VISUAL CONTENT.

2. If support exists, keep the claim.

3. If support does not exist, DELETE the claim.

4. Do not replace the deleted claim with another fact from
   pretrained knowledge.

5. Check that the claim was not created by inference from another
   lecture statement.

6. Remove inferential bridge phrases if they introduce a new
   unsupported claim.

Pay special attention to technical concepts.

Do NOT automatically expand a technical concept beyond what
the lecturer actually explained.

Only output the remaining supported information.
"""


class QwenGeneratorAdapter(IAnswerGenerator):

    def __init__(
        self,
        runtime: QwenRuntime,
    ) -> None:
        self._runtime = runtime
        self._processor = runtime.processor
        self._model = runtime.model

        logger.info(
            "qwen_generator_adapter_ready"
        )

    # ==========================================================
    # EVIDENCE SELECTION
    # ==========================================================

    def _select_evidence(
        self,
        evidence: List[RetrievedEvidence],
    ) -> List[RetrievedEvidence]:
        """
        Preserve the relevance ordering produced by the retriever.

        The retriever is responsible for determining which lecture
        sections are relevant to the question.

        The generator should not introduce another independent
        relevance-ranking strategy using arbitrary speech, visual,
        multimodal, or topic-specific weights.

        Duplicate concept IDs are removed while preserving the
        original retrieval order.
        """

        if not evidence:
            return []

        selected: List[RetrievedEvidence] = []
        selected_ids = set()

        for item in evidence:
            concept_id = item.concept.concept_id

            if concept_id in selected_ids:
                continue

            selected_ids.add(concept_id)
            selected.append(item)

        logger.info(
            "qwen_evidence_selected | "
            f"input={len(evidence)} | "
            f"selected={len(selected)}"
        )

        for index, item in enumerate(
            selected,
            start=1,
        ):
            concept = item.concept

            logger.info(
                "qwen_selected_evidence | "
                f"rank={index} | "
                f"time={concept.start_time:.1f}-"
                f"{concept.end_time:.1f}s | "
                f"semantic={item.semantic_score:.3f} | "
                f"speech={item.speech_score:.3f} | "
                f"visual={item.visual_score:.3f} | "
                f"temporal={item.temporal_score:.3f} | "
                f"speech_present="
                f"{bool(concept.speech_text.strip())} | "
                f"visual_present="
                f"{bool(concept.visual_description.strip())}"
            )

        return selected

    # ==========================================================
    # GENERATE
    # ==========================================================

    def generate(
        self,
        question: str,
        evidence: List[RetrievedEvidence],
    ) -> GeneratedAnswer:

        # ------------------------------------------------------
        # No evidence
        # ------------------------------------------------------

        if not evidence:
            logger.warning(
                "qwen_generation_skipped | reason=no_evidence"
            )

            return GeneratedAnswer(
                question=question,
                answer=(
                    "The lecture does not provide enough "
                    "information to answer this question."
                ),
                explanation=(
                    "No relevant lecture evidence was found."
                ),
                timestamp_start=None,
                timestamp_end=None,
                video_id=None,
                evidence_ids=[],
            )

        # ------------------------------------------------------
        # Select evidence
        # ------------------------------------------------------

        selected_evidence = self._select_evidence(
            evidence
        )

        if not selected_evidence:
            logger.warning(
                "qwen_generation_skipped | "
                "reason=no_selected_evidence"
            )

            return GeneratedAnswer(
                question=question,
                answer=(
                    "The lecture does not provide enough "
                    "information to answer this question."
                ),
                explanation=(
                    "The available lecture evidence could not "
                    "support a reliable answer."
                ),
                timestamp_start=None,
                timestamp_end=None,
                video_id=None,
                evidence_ids=[],
            )

        # ------------------------------------------------------
        # Build evidence block
        # ------------------------------------------------------

        evidence_block_parts = []
        seen_visual_descriptions = set()

        for i, retrieved in enumerate(
            selected_evidence,
            start=1,
        ):
            concept = retrieved.concept

            # --------------------------------------------------
            # Lecturer speech
            # --------------------------------------------------

            speech_text = (
                concept.speech_text.strip()
                if concept.speech_text
                else ""
            )

            if not speech_text:
                speech_text = (
                    "No lecturer speech available."
                )

            # --------------------------------------------------
            # Visual content
            # --------------------------------------------------

            visual_description = (
                concept.visual_description.strip()
                if concept.visual_description
                else ""
            )

            if not visual_description:
                visual_description = (
                    "No visual description available."
                )

            # --------------------------------------------------
            # Prevent exact duplicate visual descriptions
            # --------------------------------------------------

            visual_key = (
                visual_description.lower().strip()
            )

            if visual_key in seen_visual_descriptions:
                visual_description_for_prompt = (
                    "Repeated visual description; "
                    "use the earlier occurrence."
                )
            else:
                visual_description_for_prompt = (
                    visual_description
                )

                if visual_key:
                    seen_visual_descriptions.add(
                        visual_key
                    )

            # --------------------------------------------------
            # Generated metadata
            #
            # Kept only as weak context.
            # It is explicitly NOT evidence.
            # --------------------------------------------------

            multimodal_concept = (
                concept.concept.strip()
                if concept.concept
                else "Not specified."
            )

            topic = (
                concept.topic.strip()
                if concept.topic
                else "Not specified."
            )

            evidence_block_parts.append(
                f"""
--------------------------------------------------
LECTURE SECTION {i}
--------------------------------------------------
TIME:
{concept.start_time:.1f}s - {concept.end_time:.1f}s

LECTURER SPEECH:
{speech_text}

VISUAL CONTENT:
{visual_description_for_prompt}

GENERATED CONCEPT LABEL
(WEAK CONTEXT ONLY — NOT FACTUAL EVIDENCE):
{multimodal_concept}

GENERATED TOPIC LABEL
(WEAK CONTEXT ONLY — NOT FACTUAL EVIDENCE):
{topic}
"""
            )

        evidence_block = "\n".join(
            evidence_block_parts
        )

        # ------------------------------------------------------
        # Build prompt
        # ------------------------------------------------------

        prompt = _PROMPT.format(
            question=question,
            evidence_block=evidence_block,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        try:
            # --------------------------------------------------
            # Prepare Qwen input
            # --------------------------------------------------

            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self._processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(
                self._model.device
            )

            logger.info(
                "qwen_generation_started | "
                f"question_len={len(question)} | "
                f"evidence_count={len(selected_evidence)}"
            )

            # --------------------------------------------------
            # Generate
            # --------------------------------------------------

            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=220,
                    do_sample=False,
                )

            # --------------------------------------------------
            # Remove prompt tokens
            # --------------------------------------------------

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    inputs.input_ids,
                    generated_ids,
                )
            ]

            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            raw = output_text[0].strip()

            logger.info(
                f"qwen_raw_output | output={raw}"
            )

            # --------------------------------------------------
            # Parse structured output
            # --------------------------------------------------

            answer_text, explanation_text = (
                self._parse_output(raw)
            )

            logger.info(
                "qwen_parsed_answer | "
                f"answer={answer_text}"
            )

            logger.info(
                "qwen_parsed_explanation | "
                f"explanation={explanation_text}"
            )

            # --------------------------------------------------
            # Grounding metadata
            #
            # The retriever provides the relevance order.
            # Therefore the first selected item is the strongest
            # retrieved grounding section.
            #
            # We use its timestamp as the answer timestamp rather
            # than combining the full temporal span of every
            # retrieved section.
            # --------------------------------------------------

            best_evidence = selected_evidence[0]

            evidence_ids = [
                retrieved.concept.concept_id
                for retrieved in selected_evidence
            ]

            timestamp_start = (
                best_evidence.concept.start_time
            )

            timestamp_end = (
                best_evidence.concept.end_time
            )

            video_id = (
                best_evidence.concept.video_id
            )

            logger.info(
                "qwen_answer_generated | "
                f"timestamp={timestamp_start:.1f}-"
                f"{timestamp_end:.1f}s | "
                f"evidence_count={len(evidence_ids)}"
            )

            return GeneratedAnswer(
                question=question,
                answer=answer_text,
                explanation=explanation_text,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                video_id=video_id,
                evidence_ids=evidence_ids,
            )

        except Exception as exc:
            logger.exception(
                f"qwen_generation_failed | error={exc}"
            )

            raise RuntimeError(
                f"Qwen answer generation failed: {exc}"
            ) from exc

    # ==========================================================
    # PARSE QWEN STRUCTURED OUTPUT
    # ==========================================================

    def _parse_output(
        self,
        text: str,
    ) -> tuple[str, str]:
        """
        Parse Qwen's structured response.

        Supports both:

            ANSWER:
            <answer>

            EXPLANATION:
            <explanation>

        and Markdown-formatted headers:

            **ANSWER:**
            <answer>

            **EXPLANATION:**
            <explanation>

        Qwen sometimes adds Markdown emphasis even though the
        prompt requests plain section headers, so the parser
        normalizes those headers before extracting the sections.
        """

        if not text:
            return (
                "",
                "The answer is based on the available "
                "lecture evidence.",
            )

        # ------------------------------------------------------
        # Normalize the raw output
        # ------------------------------------------------------

        normalized_text = text.strip()

        normalized_text = re.sub(
            r"(?im)^\s*\*{1,3}\s*ANSWER\s*:\s*\*{1,3}\s*$",
            "ANSWER:",
            normalized_text,
        )

        normalized_text = re.sub(
            r"(?im)^\s*\*{1,3}\s*EXPLANATION\s*:\s*\*{1,3}\s*$",
            "EXPLANATION:",
            normalized_text,
        )

        normalized_text = re.sub(
            r"(?im)^\s*ANSWER\s*:\s*$",
            "ANSWER:",
            normalized_text,
        )

        normalized_text = re.sub(
            r"(?im)^\s*EXPLANATION\s*:\s*$",
            "EXPLANATION:",
            normalized_text,
        )

        # ------------------------------------------------------
        # Extract ANSWER section
        # ------------------------------------------------------

        answer_match = re.search(
            r"(?is)\bANSWER\s*:\s*(.*?)(?=\bEXPLANATION\s*:|$)",
            normalized_text,
        )

        # ------------------------------------------------------
        # Extract EXPLANATION section
        # ------------------------------------------------------

        explanation_match = re.search(
            r"(?is)\bEXPLANATION\s*:\s*(.*)$",
            normalized_text,
        )

        answer = (
            answer_match.group(1).strip()
            if answer_match
            else ""
        )

        explanation = (
            explanation_match.group(1).strip()
            if explanation_match
            else ""
        )

        # ------------------------------------------------------
        # Remove accidental Markdown emphasis around content
        # ------------------------------------------------------

        answer = answer.strip("* \n\t")
        explanation = explanation.strip("* \n\t")

        # ------------------------------------------------------
        # Fallback
        # ------------------------------------------------------

        if not answer:
            answer = normalized_text.strip()

            if answer.upper() in {
                "ANSWER:",
                "**ANSWER:**",
            }:
                answer = ""

        if not answer:
            answer = (
                "The lecture does not provide enough "
                "information to answer this question."
            )

        if not explanation:
            explanation = (
                "The answer is based on the available "
                "lecture evidence."
            )

        return answer, explanation