import numpy as np


def getSpeaking(pred_arr, minLength, fps):
    # Obtains the sequences where a person is talking during
    # more than minLength consecutive frames
    prev_idx = 0
    idx_list = []
    for i, num in enumerate(pred_arr):
        if num == 1:
            # Check if this is the end of a sequence
            if i == len(pred_arr) - 1 or pred_arr[i+1] == 0:
                if i-prev_idx + 1 >= minLength:  # +1 because the sequence includes the current index
                    # +1 because the sequence includes the current index
                    idx_list.append((prev_idx/fps, (i+1)/fps))
                prev_idx = i + 1  # Move to the next index
        else:
            if i-prev_idx >= minLength:
                idx_list.append((prev_idx/fps, i/fps))
            prev_idx = i + 1  # Move to the next index
    return idx_list


def getSpeaking2(pred_arr, minLength, max_zeros, fps):
    # Obtains the sequences where a person is talking during
    # more than minLength consecutive frames
    prev_idx = 0
    last_one_idx = 0
    zero_count = 0
    idx_list = []
    for i, num in enumerate(pred_arr):
        if num == 1:
            # Append end of array if current is last element
            if i == len(pred_arr) - 1:
                if i-prev_idx + 1 >= minLength:
                    idx_list.append((prev_idx/fps, (i+1)/fps))
                prev_idx = i + 1  # Move to the next index

            # Start sequence with ones always
            if prev_idx <= i and zero_count != 0:
                prev_idx = i

            # Reset zero counter and add last 1 idx
            zero_count = 0
            last_one_idx = i
        else:
            # zero
            zero_count += 1
            # If zeroes are at start of sequence
            if last_one_idx == 0 or last_one_idx == prev_idx+1:
                prev_idx = i + 1

            if zero_count == max_zeros:
                # Check sequence length
                if i-prev_idx >= minLength:
                    idx_list.append((prev_idx/fps, (i-max_zeros+1)/fps))
                prev_idx = i + 1  # Move to the next index

    return idx_list


# (2, 12), (21, 25)
array = [
    0, 0,  # 2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 12
    0, 0, 0, 0, 0, 0, 0, 0,  # 21
    1, 1, 1, 1, 1  # 25
]

idx_list = getSpeaking(array, 5, 25)
print("original", idx_list)
idx_list = getSpeaking2(array, 5, 5, 25)
print("with zeros:", idx_list)
