"""
lecture_memory/adapters/test_simple_event_former.py
---------------------------------------------------
Comprehensive test suite for SimpleVisualEventFormer.

Can be run via:
    pytest lecture_memory/adapters/test_simple_event_former.py
or directly:
    python lecture_memory/adapters/test_simple_event_former.py
"""

from __future__ import annotations

import sys
import unittest
from typing import List

from lecture_memory.adapters.simple_event_former import SimpleVisualEventFormer
from lecture_memory.domain.models import SpeechSegment, VisualEvidence, VisualEvent


class TestSimpleVisualEventFormer(unittest.TestCase):
    """Unit and functional tests for SimpleVisualEventFormer."""

    def setUp(self) -> None:
        self.former = SimpleVisualEventFormer()
        self.video_id = "test_video_101"

    # =========================================================================
    # 1. Fallback & Edge Cases
    # =========================================================================

    def test_visual_only_fallback_when_no_speech(self) -> None:
        """When speech_segments is empty, should generate events from visual evidence."""
        visuals = [
            VisualEvidence(
                visual_id="ve_1",
                frame_id="f_1",
                video_id=self.video_id,
                timestamp=10.0,
                caption="A diagram of OSI architecture.",
            ),
            VisualEvidence(
                visual_id="ve_2",
                frame_id="f_2",
                video_id=self.video_id,
                timestamp=25.0,
                caption="Data link frame header breakdown.",
            ),
        ]

        events = self.former.form_events(
            video_id=self.video_id,
            visual_evidence=visuals,
            speech_segments=[],
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].start_time, 10.0)
        self.assertEqual(events[0].end_time, 10.0)
        self.assertEqual(events[0].frame_ids, ["f_1"])
        self.assertIn("OSI architecture", events[0].description)

        self.assertEqual(events[1].start_time, 25.0)
        self.assertEqual(events[1].frame_ids, ["f_2"])

    def test_single_speech_segment(self) -> None:
        """A single speech segment should produce exactly one unified event."""
        units = [
            SpeechSegment(
                segment_id="s_1",
                video_id=self.video_id,
                start_time=0.0,
                end_time=15.0,
                text="Welcome to computer networks overview.",
            )
        ]
        visuals = [
            VisualEvidence(
                visual_id="ve_1",
                frame_id="f_1",
                video_id=self.video_id,
                timestamp=5.0,
                caption="Title slide showing Computer Networks.",
            )
        ]

        events = self.former.form_events(
            video_id=self.video_id,
            visual_evidence=visuals,
            speech_segments=units,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_time, 0.0)
        self.assertEqual(events[0].end_time, 15.0)
        self.assertEqual(events[0].frame_ids, ["f_1"])
        self.assertIn("networks overview", events[0].title.lower())

    # =========================================================================
    # 2. Explicit Discourse Transitions
    # =========================================================================

    def test_explicit_discourse_transitions(self) -> None:
        """Transitions like 'Now', 'Next', 'Moving on' should trigger new events."""
        units = [
            SpeechSegment("s_1", self.video_id, 0.0, 10.0, "We begin with the physical layer characteristics."),
            SpeechSegment("s_2", self.video_id, 10.0, 20.0, "Signals propagate across copper or fiber."),
            SpeechSegment("s_3", self.video_id, 20.0, 30.0, "Now let's examine the data link layer framing."),
            SpeechSegment("s_4", self.video_id, 30.0, 40.0, "Frames contain MAC addresses and checksums."),
            SpeechSegment("s_5", self.video_id, 40.0, 50.0, "Moving on to packet forwarding in IP networks."),
        ]

        events = self.former.form_events(
            video_id=self.video_id,
            visual_evidence=[],
            speech_segments=units,
        )

        self.assertEqual(len(events), 3)
        # Event 1: physical layer (0s - 20s)
        self.assertEqual(events[0].start_time, 0.0)
        self.assertEqual(events[0].end_time, 20.0)
        # Event 2: data link layer (20s - 40s)
        self.assertEqual(events[1].start_time, 20.0)
        self.assertEqual(events[1].end_time, 40.0)
        # Event 3: moving on to packet forwarding (40s - 50s)
        self.assertEqual(events[2].start_time, 40.0)
        self.assertEqual(events[2].end_time, 50.0)

    # =========================================================================
    # 3. Bridge Transitions
    # =========================================================================

    def test_bridge_transition_boundary_placement(self) -> None:
        """
        'That brings us to...' announces the next topic,
        so the boundary should be placed AFTER the bridge sentence.
        """
        units = [
            SpeechSegment("s_1", self.video_id, 0.0, 10.0, "That covers standard checksum computation."),
            SpeechSegment("s_2", self.video_id, 10.0, 20.0, "That brings us to the transport layer protocols."),
            SpeechSegment("s_3", self.video_id, 20.0, 30.0, "The Transmission Control Protocol provides reliable byte streams."),
            SpeechSegment("s_4", self.video_id, 30.0, 40.0, "It manages flow control via sliding windows."),
        ]

        events = self.former.form_events(
            video_id=self.video_id,
            visual_evidence=[],
            speech_segments=units,
        )

        self.assertEqual(len(events), 2)
        # Event 1 should absorb the bridge sentence (0s - 20s)
        self.assertEqual(events[0].start_time, 0.0)
        self.assertEqual(events[0].end_time, 20.0)
        self.assertIn("transport layer protocols", events[0].description)

        # Event 2 starts where substantive explanation begins (20s - 40s)
        self.assertEqual(events[1].start_time, 20.0)
        self.assertEqual(events[1].end_time, 40.0)
        self.assertIn("Transmission Control Protocol", events[1].description)

    # =========================================================================
    # 4. Structural Markers
    # =========================================================================

    def test_structural_markers(self) -> None:
        """'Section 2', 'The second stage', etc. should trigger boundaries when separated by content."""
        units = [
            SpeechSegment("s_1", self.video_id, 0.0, 10.0, "The first section covers introductory terminology."),
            SpeechSegment("s_2", self.video_id, 10.0, 20.0, "Here we define the core network components."),
            SpeechSegment("s_3", self.video_id, 20.0, 30.0, "Section two discusses transmission media."),
            SpeechSegment("s_4", self.video_id, 30.0, 40.0, "Guided media includes twisted pair and coaxial cables."),
            SpeechSegment("s_5", self.video_id, 40.0, 50.0, "Part three explores error detection algorithms."),
            SpeechSegment("s_6", self.video_id, 50.0, 60.0, "Parity checks and cyclic redundancy checks are implemented."),
        ]

        events = self.former.form_events(
            video_id=self.video_id,
            visual_evidence=[],
            speech_segments=units,
        )

        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].start_time, 0.0)
        self.assertEqual(events[0].end_time, 20.0)
        self.assertEqual(events[1].start_time, 20.0)
        self.assertEqual(events[1].end_time, 40.0)
        self.assertEqual(events[2].start_time, 40.0)
        self.assertEqual(events[2].end_time, 60.0)

    # =========================================================================
    # 5. Continuations & Conclusions (No Split)
    # =========================================================================

    def test_continuations_do_not_split(self) -> None:
        """Sentences starting with 'And', 'So', 'Because', 'For example' should not split."""
        units = [
            SpeechSegment("s_1", self.video_id, 0.0, 10.0, "Let us review packet dropping."),
            SpeechSegment("s_2", self.video_id, 10.0, 20.0, "Because buffer queues can overflow under heavy load."),
            SpeechSegment("s_3", self.video_id, 20.0, 30.0, "So routers discard unserviced datagrams."),
            SpeechSegment("s_4", self.video_id, 30.0, 40.0, "For example, RED algorithm monitors average queue length."),
            SpeechSegment("s_5", self.video_id, 40.0, 50.0, "In other words, congestion collapse is mitigated."),
        ]

        events = self.former.form_events(
            video_id=self.video_id,
            visual_evidence=[],
            speech_segments=units,
        )

        # All of these are continuations of the same topic
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_time, 0.0)
        self.assertEqual(events[0].end_time, 50.0)

    def test_lecture_conclusion_does_not_split(self) -> None:
        """'In conclusion', 'To conclude' should remain part of the current event."""
        units = [
            SpeechSegment("s_1", self.video_id, 0.0, 15.0, "We now summarize congestion avoidance mechanisms."),
            SpeechSegment("s_2", self.video_id, 15.0, 30.0, "Additive increase and multiplicative decrease are applied."),
            SpeechSegment("s_3", self.video_id, 30.0, 45.0, "In conclusion, TCP achieves fair bandwidth sharing."),
        ]

        events = self.former.form_events(
            video_id=self.video_id,
            visual_evidence=[],
            speech_segments=units,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].end_time, 45.0)

    # =========================================================================
    # 6. Outro & Sponsor Detection
    # =========================================================================

    def test_outro_section_consolidation(self) -> None:
        """Outro phrases should create an outro event and avoid micro-splitting."""
        units = [
            SpeechSegment("s_1", self.video_id, 0.0, 20.0, "Let us look at graph search algorithms."),
            SpeechSegment("s_2", self.video_id, 20.0, 40.0, "Dijkstra finds shortest paths in weighted graphs."),
            SpeechSegment("s_3", self.video_id, 40.0, 50.0, "Thanks for watching everyone."),
            SpeechSegment("s_4", self.video_id, 50.0, 60.0, "Please like and subscribe if you enjoyed this."),
            SpeechSegment("s_5", self.video_id, 60.0, 70.0, "And now a quick word from our sponsor."),
        ]

        events = self.former.form_events(
            video_id=self.video_id,
            visual_evidence=[],
            speech_segments=units,
        )

        # Event 1: Dijkstra lecture (0s - 40s)
        # Event 2: Consolidated outro (40s - 70s)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].start_time, 0.0)
        self.assertEqual(events[0].end_time, 40.0)

        self.assertEqual(events[1].start_time, 40.0)
        self.assertEqual(events[1].end_time, 70.0)
        self.assertIn("like and subscribe", events[1].description)
        self.assertIn("quick word from our sponsor", events[1].description)

    # =========================================================================
    # 7. Boundary Cleanup & Redundancy
    # =========================================================================

    def test_redundant_adjacent_boundaries_cleanup(self) -> None:
        """Adjacent boundary indices should be collapsed to avoid micro-events."""
        boundaries = [2, 3, 7, 8, 12]
        units = [SpeechSegment(f"s_{i}", self.video_id, i * 5.0, (i + 1) * 5.0, f"segment text {i}") for i in range(15)]

        cleaned = self.former._remove_redundant_boundaries(boundaries, units)
        self.assertEqual(cleaned, [2, 7, 12])

    # =========================================================================
    # 8. Title Generation
    # =========================================================================

    def test_title_strips_leading_filler_words(self) -> None:
        """Titles should strip leading conversational fillers like 'and', 'so', 'now'."""
        units = [
            SpeechSegment("s_1", self.video_id, 0.0, 10.0, "And now next we inspect pipeline throughput metrics in detail."),
        ]
        title = self.former._build_title(units)
        # Should start with "we inspect pipeline throughput"
        self.assertTrue(title.startswith("we inspect pipeline throughput") or "pipeline throughput" in title)
        self.assertFalse(title.startswith("And now next"))

    def test_title_fallback_on_short_text(self) -> None:
        """Very short speech text should fallback to default title."""
        units = [SpeechSegment("s_1", self.video_id, 0.0, 5.0, "Hi.")]
        title = self.former._build_title(units)
        self.assertEqual(title, "Lecture episode")

    # =========================================================================
    # 9. Visual Evidence Correlation
    # =========================================================================

    def test_visual_evidence_temporal_mapping(self) -> None:
        """Only visual frames falling within the event time window should be attached."""
        units = [
            SpeechSegment("s_1", self.video_id, 0.0, 20.0, "Let us examine sorting algorithms."),
            SpeechSegment("s_2", self.video_id, 20.0, 40.0, "Moving on to quicksort partitioning."),
        ]
        visuals = [
            VisualEvidence("ve_1", "f_1", self.video_id, 5.0, "Bubble sort slide"),
            VisualEvidence("ve_2", "f_2", self.video_id, 15.0, "Insertion sort slide"),
            VisualEvidence("ve_3", "f_3", self.video_id, 25.0, "Quicksort pivot selection"),
            VisualEvidence("ve_4", "f_4", self.video_id, 55.0, "Out of bounds frame"),  # Beyond speech duration
        ]

        events = self.former.form_events(
            video_id=self.video_id,
            visual_evidence=visuals,
            speech_segments=units,
        )

        self.assertEqual(len(events), 2)
        # Event 1 (0.0s - 20.0s) should have f_1, f_2
        self.assertEqual(events[0].frame_ids, ["f_1", "f_2"])
        # Event 2 (20.0s - 40.0s) should have f_3
        self.assertEqual(events[1].frame_ids, ["f_3"])


# =============================================================================
# Direct Runner
# =============================================================================

def run_tests() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSimpleVisualEventFormer)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
