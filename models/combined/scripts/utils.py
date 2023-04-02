import numpy as np

def parse_line_only_coref(line):
    d = {
            "act": [],
            "slots": [],
            "request_slots": [],
            "objects": [],
        }

    splits = line.split('<EOB>')
    splits = splits[0].split("[  ] ()")
    splits = splits[1].split("< ")
    splits = splits[1].split(" >")
    splits = splits[0].split(", ")

    for item in splits:
        try:
            d['objects'].append(int(item))
        except:
            pass

    return [d]


def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0
    prec = n_correct / n_pred if n_pred != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

    return rec, prec, f1


def d_f1(n_correct, n_true, n_pred):
    # 1/r + 1/p = 2/F1
    # dr / r^2 + dp / p^2 = 2dF1 /F1^2
    # dF1 = 1/2 F1^2 (dr/r^2 + dp/p^2)
    dr = b_stderr(n_true, n_correct)
    dp = b_stderr(n_pred, n_correct)

    r = n_correct / n_true
    p = n_correct / n_pred
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0

    d_f1 = 0.5 * f1 ** 2 * (dr / r ** 2 + dp / p ** 2)
    return d_f1


def b_stderr(n_total, n_pos):
    return np.std(b_arr(n_total, n_pos)) / np.sqrt(n_total)


def b_arr(n_total, n_pos):
    out = np.zeros(int(n_total))
    out[: int(n_pos)] = 1.0
    return out