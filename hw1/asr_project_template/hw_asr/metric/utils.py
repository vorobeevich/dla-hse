import editdistance

def calc_cer(target_text: str, predicted_text: str) -> float:
    if len(target_text) == 0:
        return 1
    return editdistance.distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    splitted_target = target_text.split()
    if len(splitted_target) == 0:
        return 1
    return editdistance.distance(splitted_target, predicted_text.split()) / len(splitted_target)