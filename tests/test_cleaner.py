from text_cleaner import cleaner


text = "Nazywam się Jan Kowalski. Mój email to abc123@gmail.com, mój numer telefonu to 123456789. Mieszkam w Polsce, w Warszawie."


def test_anonymize_text():
    assert cleaner.anonymize_text(text, cleaner.Engine.PL) == "Nazywam się [private]. Mój email to [private], mój numer telefonu to [private]. Mieszkam w Polsce, w Warszawie."