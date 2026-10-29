"""Rule-based tidying of Model 1 v3 notes (src/model1_notes/postprocess.py)."""

from src.config import load_config
from src.model1_notes.notes import Note, render
from src.model1_notes.postprocess import postprocess

CFG = load_config("configs/model1_v3.yaml")["inference"]["postprocess"]


def meeting(rows):
    """``(speaker, text, zone, kept)`` rows -> notes."""
    return [Note(i, s, t, z, 0.9 if k else 0.1, k) for i, (s, t, z, k) in enumerate(rows)]


def test_confirmation_attaches_and_gives_owner_and_deadline():
    notes = meeting(
        [
            ("Maria", "Priya, can you book three sessions?", "action_item", True),
            ("Priya", "Yes, I will book them by Friday.", "discussion", False),
            ("Tom", "Sounds good.", "discussion", False),
        ]
    )
    notes[1].deadline = "Friday"
    postprocess(notes, [-1, -1, -1], CFG)
    request = notes[0]
    assert [r.index for r in request.followups] == [1, 2]
    assert request.owner == "Priya" and request.deadline == "Friday"  # only the commitment
    assert "↳ Priya: Yes, I will book them by Friday." in render([request])


def test_confirmation_uses_the_depends_on_pointer_and_skips_same_speaker():
    notes = meeting(
        [
            ("Anna", "We launch on the 15th.", "decision", True),
            ("Ben", "The bug is fixed.", "discussion", True),
            ("Chloe", "Okay, agreed.", "discussion", False),
            ("Anna", "Yes.", "discussion", False),
        ]
    )
    postprocess(notes, [-1, -1, 0, 0], CFG)  # Chloe and Anna both point at sentence 0
    assert [r.index for r in notes[0].followups] == [2]  # Anna does not confirm herself
    assert notes[1].followups == []


def test_long_or_unrelated_replies_are_not_attached():
    notes = meeting(
        [
            ("Anna", "Ben, please fix the payment bugs.", "action_item", True),
            (
                "Ben",
                "Okay but first I need to understand why the tests on the staging "
                "server keep failing every night.",
                "discussion",
                False,
            ),
            ("Chloe", "The weather is nice.", "discussion", False),
        ]
    )
    postprocess(notes, [-1, -1, -1], CFG)
    assert notes[0].followups == [] and notes[0].owner is None


def test_first_person_action_item_gets_its_speaker_as_owner():
    notes = meeting([("Priya", "I will prepare the test files by Wednesday.", "action_item", True)])
    postprocess(notes, [-1], CFG)
    assert notes[0].owner == "Priya"
    named = meeting([("Maria", "Priya will book the rooms.", "action_item", True)])
    assert postprocess(named, [-1], CFG)[0].owner is None  # not first person: left to the model
    assert (
        postprocess(meeting([("A", "I will.", "action_item", True)]), [-1], None)[0].owner is None
    )


def test_bare_commitment_does_not_join_another_replys_chain():
    notes = meeting(
        [
            ("Anna", "Ben, please fix the payment bugs by Friday.", "action_item", True),
            ("Ben", "Okay, I'll send an update on Wednesday.", "discussion", False),
            ("Chloe", "I'll book the newsletter for the 15th.", "discussion", False),
            ("Anna", "Okay, great.", "discussion", False),
        ]
    )
    postprocess(notes, [-1, -1, -1, -1], CFG)
    assert [r.index for r in notes[0].followups] == [1]  # Chloe's own task is not Ben's reply


def test_reply_skips_an_unkept_fragment_to_reach_the_request():
    notes = meeting(
        [
            ("a", "And ask him about the rent.", "action_item", True),
            ("a", "increase, too.", "discussion", False),
            ("b", "OK, I will email him tonight.", "discussion", False),
        ]
    )
    notes[2].deadline = "tonight"
    postprocess(notes, [-1, -1, -1], CFG)
    assert [r.index for r in notes[0].followups] == [2]
    assert notes[0].owner == "b" and notes[0].deadline == "tonight"


def test_untrained_pointer_is_ignored():
    notes = meeting(
        [
            ("a", "Ask him about the rent increase.", "action_item", True),
            ("b", "The weather is nice.", "discussion", False),
            ("b", "OK, I will email him tonight.", "discussion", False),
        ]
    )
    postprocess(notes, [-1, -1, 1], {**CFG, "use_pointer": False})  # pointer says sentence 1
    assert [r.index for r in notes[0].followups] == [2]


def test_reply_does_not_take_the_owner_of_a_first_person_commitment():
    notes = meeting(
        [
            ("a", "I can work from home that day.", "action_item", True),
            ("b", "Great, then I will tell him to come at ten.", "action_item", False),
        ]
    )
    postprocess(notes, [-1, -1], CFG)
    assert notes[0].owner == "a"
