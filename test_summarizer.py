from core import summarizer

text = (
    "This is a test transcript. It contains several sentences. The meeting covered project scope, "
    "timelines, and deliverables. Action items were assigned to the team. Follow up next week."
)
print('Title:', summarizer.generate_title(text))
print('\nSummary:\n', summarizer.summarize(text))
