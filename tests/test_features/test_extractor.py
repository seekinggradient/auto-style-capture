from auto_style_capture.features.extractor import extract_features, extract_corpus_features
from auto_style_capture.features.lexical import extract_lexical_features
from auto_style_capture.features.syntactic import extract_syntactic_features
from auto_style_capture.features.punctuation import extract_punctuation_features
from auto_style_capture.features.readability import extract_readability_features


def test_extract_features(sample_text):
    profile = extract_features(sample_text)
    assert len(profile.features) > 0
    assert "lex_ttr" in profile.features
    assert "syn_sent_length_mean" in profile.features
    assert "punct_comma_rate" in profile.features
    assert "read_flesch_kincaid" in profile.features


def test_extract_corpus_features(sample_text):
    profile = extract_corpus_features([sample_text, sample_text])
    assert len(profile.features) > 0


def test_lexical_features(sample_text):
    features = extract_lexical_features(sample_text)
    assert 0 < features["lex_ttr"] <= 1
    assert features["lex_mean_word_length"] > 0
    assert features["lex_yules_k"] >= 0


def test_syntactic_features(sample_text):
    features = extract_syntactic_features(sample_text)
    assert features["syn_sent_length_mean"] > 0
    assert features["syn_total_sentences"] > 0


def test_punctuation_features(sample_text):
    features = extract_punctuation_features(sample_text)
    assert features["punct_period_rate"] > 0
    assert features["punct_density"] > 0


def test_readability_features(sample_text):
    features = extract_readability_features(sample_text)
    assert "read_flesch_kincaid" in features
    assert "read_gunning_fog" in features


def test_profile_to_vector(sample_text):
    profile = extract_features(sample_text)
    vec = profile.to_vector()
    assert len(vec) == len(profile.features)


def test_profile_compare(sample_text):
    p1 = extract_features(sample_text)
    p2 = extract_features("Short. Very short. Yes.")
    comparison = p1.compare(p2)
    assert "real=" in comparison
    assert "gen=" in comparison


def test_empty_text():
    profile = extract_features("")
    assert isinstance(profile.features, dict)
