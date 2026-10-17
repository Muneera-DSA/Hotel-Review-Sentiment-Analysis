from hotel_sentiment.preprocess import normalise, strip_placeholder


def test_negations_are_kept():
    # The original version removed "not" via NLTK stopwords, flipping meaning.
    assert normalise("The room was NOT clean") == "the room was not clean"
    assert "not" in normalise("Not bad at all").split()


def test_contractions_expand_to_not():
    assert normalise("It wasn't clean") == "it was not clean"
    assert normalise("Staff didn’t help") == "staff did not help"
    assert normalise("We can't complain") == "we can not complain"
    assert normalise("It won't happen again") == "it will not happen again"


def test_contractions_with_stripped_apostrophes():
    # The Kaggle file stores "wasn't" as "wasn t"; the negation must survive.
    assert normalise("It wasn t clean") == "it was not clean"
    assert normalise("I can t sleep") == "i can not sleep"
    assert normalise("We won t return") == "we will not return"
    assert normalise("Don t go") == "do not go"
    assert normalise("Tea wasn t free") == "tea was not free"


def test_non_letters_and_whitespace():
    assert normalise("  Great!!! 10/10   value  ") == "great value"
    assert normalise(None) == ""


def test_placeholders_removed_but_real_text_kept():
    assert strip_placeholder(" No Negative ") == ""
    assert strip_placeholder("No Positive") == ""
    assert strip_placeholder(" Nothing to complain about ") == "Nothing to complain about"
    assert strip_placeholder(float("nan")) == ""
