from nlp.schema import VERDICTS


def test_verdicts_exact():
    assert VERDICTS == ["model right", "label right", "both OK", "both wrong", "ambiguous"]
