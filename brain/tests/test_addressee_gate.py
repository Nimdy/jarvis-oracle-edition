"""Tests for the AddresseeGate — directedness / turn-ownership classifier."""

import pytest
from perception.addressee import AddresseeGate, AddresseeResult


class TestExplicitNegation:
    """Explicit 'not talking to you' patterns must suppress."""

    @pytest.fixture(autouse=True)
    def gate(self):
        self.gate = AddresseeGate()

    def test_not_talking_to_you(self):
        r = self.gate.check("I'm not talking to you")
        assert not r.addressed
        assert r.suppressed
        assert r.reason == "explicit_negation"
        assert r.confidence >= 0.9

    def test_disregard_that(self):
        r = self.gate.check("disregard that")
        assert r.suppressed
        assert r.reason == "explicit_negation"

    def test_dont_listen_to_that(self):
        r = self.gate.check("Jarvis, don't listen to that")
        assert r.suppressed
        assert r.reason == "explicit_negation"

    def test_stop_listening_overrides_room_transcript(self):
        text = (
            "You makin' felt? Have you ever seen the dryer look like this? "
            "Jarvis, don't listen to that. It's felt. That's how you make felt. "
            "Never change the l- Jarvis, stop listening."
        )
        r = self.gate.check(text, had_wake_word=True)
        assert not r.addressed
        assert r.suppressed
        assert r.reason == "explicit_negation"

    def test_talking_to_someone_else(self):
        r = self.gate.check("I was talking to someone else, not you")
        assert r.suppressed
        assert r.reason == "explicit_negation"

    def test_wasnt_referring_to_you(self):
        r = self.gate.check("I wasn't referring to you")
        assert r.suppressed
        assert r.reason == "explicit_negation"

    def test_forget_that(self):
        r = self.gate.check("Forget that, pretend you didn't hear that")
        assert r.suppressed

    def test_negation_case_insensitive(self):
        r = self.gate.check("NOT TALKING TO YOU")
        assert r.suppressed

    def test_none_of_your_business(self):
        r = self.gate.check("That's none of your business")
        assert r.suppressed


class TestDismissiveComplaints:
    """Post-response complaints should suppress, not create new work."""

    @pytest.fixture(autouse=True)
    def gate(self):
        self.gate = AddresseeGate()

    def test_waste_of_time(self):
        r = self.gate.check("That was a waste of time")
        assert r.suppressed
        assert r.reason == "dismissive_complaint"

    def test_who_asked(self):
        r = self.gate.check("Nobody asked you")
        assert r.suppressed

    def test_everything_you_said(self):
        r = self.gate.check("Everything you said was useless")
        assert r.suppressed

    def test_i_didnt_ask(self):
        r = self.gate.check("I didn't ask you")
        assert r.suppressed


class TestPositiveAddressing:
    """Commands and name mentions should be recognized as addressed."""

    @pytest.fixture(autouse=True)
    def gate(self):
        self.gate = AddresseeGate()

    def test_jarvis_name(self):
        r = self.gate.check("Jarvis, what time is it?")
        assert r.addressed
        assert r.reason == "name_mention"
        assert r.confidence >= 0.9

    def test_command_framing(self):
        r = self.gate.check("Can you tell me the weather?")
        assert r.addressed
        assert r.reason == "command_framing"

    def test_follow_up_addressed(self):
        r = self.gate.check("and what about tomorrow?", is_follow_up=True)
        assert r.addressed
        assert r.reason == "follow_up_conversation"

    def test_wake_word_default(self):
        r = self.gate.check("the sky is blue", had_wake_word=True)
        assert r.addressed
        assert r.reason == "wake_word_default"

    def test_wake_word_with_you(self):
        r = self.gate.check("what do you think about this?", had_wake_word=True)
        assert r.addressed
        assert r.reason == "wake_word_second_person"

    def test_no_wake_no_follow_up_not_addressed(self):
        r = self.gate.check("the sky is blue", had_wake_word=False, is_follow_up=False)
        assert not r.addressed


class TestStats:
    def test_counters(self):
        gate = AddresseeGate()
        gate.check("Hello Jarvis")
        gate.check("Not talking to you")
        gate.check("What time is it?")
        stats = gate.get_stats()
        assert stats["total_checked"] == 3
        assert stats["total_suppressed"] == 1
        assert stats["misaddressed_count"] == 1
