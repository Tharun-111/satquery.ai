from models.vqa.vqa_model import create_vqa_model

__all__ = ["answer_question"]

_model = None


def answer_question(image_path, question):
    global _model

    if _model is None:
        _model = create_vqa_model()

    return _model.answer(image_path, question)