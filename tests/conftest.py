import pytest

from auto_style_capture.corpus.models import Corpus, Document


@pytest.fixture
def sample_text():
    return (
        "The old man sat by the river. He watched the water move. "
        "It was cold and the wind blew hard across the plain. "
        "He did not speak. There was nothing to say about it. "
        "The fish had not come and would not come. He knew this.\n\n"
        "In the morning he would try again. The boy would be there. "
        "They would go out in the boat and the sun would be hot on the water. "
        "He thought about the lions on the beach."
    )


@pytest.fixture
def sample_corpus(sample_text):
    return Corpus(documents=[
        Document(text=sample_text, source="test1.txt"),
        Document(
            text="She walked through the garden slowly. The flowers were in bloom. "
            "Red and yellow and white. She picked one and held it to her face. "
            "It smelled like summer. She remembered summers long ago.",
            source="test2.txt",
        ),
    ])
