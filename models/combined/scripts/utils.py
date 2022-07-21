
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