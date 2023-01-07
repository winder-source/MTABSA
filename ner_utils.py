class NerUtils:
    # padding label：从0开始
    label_list = ['O', 'B-ASP', 'I-ASP', '[CLS]', '[SEP]']
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    num_labels = len(label_list) + 1  # 加上padding的标签

