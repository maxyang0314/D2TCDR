import csv
import math
import numpy as np
from collections import defaultdict
import pandas as pd


def process_csv(
    input_file,
    train_file,
    val_file,
    test_file,
    target_length=8,
    history_length=16
):

    user_sequences = defaultdict(list)

    item_to_int = {}
    item_id_counter = 1

    # ============================================================
    # 1. 讀取資料
    # ============================================================

    with open(input_file, 'r') as file:

        reader = csv.reader(file)

        for row in reader:

            user_id, item_id, rating, timestamp = row

            if int(float(rating)) >= 4:

                user_sequences[user_id].append(
                    (item_id, int(timestamp))
                )

    # ============================================================
    # 2. 按 timestamp 排序
    # ============================================================

    formatted_sequences = {}

    for user_id, interactions in user_sequences.items():

        sorted_interactions = sorted(
            interactions,
            key=lambda x: x[1]
        )

        items = [
            item
            for item, _ in sorted_interactions
        ]

        if len(items) >= 5:

            formatted_sequences[user_id] = items

    # ============================================================
    # 3. 建立 training samples
    #
    # history_length = 16
    # target_length  = 8
    #
    # 例如：
    #
    # 原始：
    # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    #
    # history_seq：
    # [1,2,3,...,16]
    #
    # seq：
    # [9,10,11,12,13,14,15,16]
    #
    # next：
    # 17
    # ============================================================

    processed_data = []

    for user_id, full_seq in formatted_sequences.items():

        # --------------------------------------------------------
        # 至少需要：
        # history_length 個 history
        # + 1 個 next
        # --------------------------------------------------------

        if len(full_seq) < 2:
            continue

        # --------------------------------------------------------
        # 建立 item ID mapping
        # --------------------------------------------------------

        for item_id in full_seq:

            if item_id not in item_to_int:

                item_to_int[item_id] = item_id_counter

                item_id_counter += 1

        # --------------------------------------------------------
        # 轉換成 integer item ID
        # --------------------------------------------------------

        full_seq = [
            item_to_int[item_id]
            for item_id in full_seq
        ]

        # --------------------------------------------------------
        # next = 最後一個 interaction
        # --------------------------------------------------------

        target_item = full_seq[-1]

        # --------------------------------------------------------
        # history = next 前面的所有 interaction
        # --------------------------------------------------------

        history = full_seq[:-1]

        # --------------------------------------------------------
        # 最多取最近 history_length 個 history
        # --------------------------------------------------------

        history_seq = history[-history_length:]

        # --------------------------------------------------------
        # model 原本使用的 seq
        # 只取 history_seq 最後 target_length 個
        # --------------------------------------------------------

        input_seq = history_seq[-target_length:]

        seq_length = len(input_seq)

        # --------------------------------------------------------
        # seq padding
        # --------------------------------------------------------

        if seq_length < target_length:

            input_seq = (
                [0] * (target_length - seq_length)
                + input_seq
            )

        # --------------------------------------------------------
        # history_seq padding
        #
        # 固定成 history_length
        # --------------------------------------------------------

        history_length_actual = len(history_seq)

        if history_length_actual < history_length:

            history_seq = (
                [0] * (history_length - history_length_actual)
                + history_seq
            )

        # --------------------------------------------------------
        # 儲存
        # --------------------------------------------------------

        processed_data.append(
            (
                input_seq,
                history_seq,
                seq_length,
                target_item
            )
        )

    print("Number of items:", len(item_to_int))

    # ============================================================
    # 4. Shuffle
    # ============================================================

    np.random.shuffle(processed_data)

    train_size = math.floor(
        len(processed_data) * 0.8
    )

    val_size = math.floor(
        len(processed_data) * 0.1
    )

    train_data = processed_data[:train_size]

    val_data = processed_data[
        train_size:
        train_size + val_size
    ]

    test_data = processed_data[
        train_size + val_size:
    ]

    # ============================================================
    # 5. 儲存
    # ============================================================

    def save_to_file(data, file_path):

        df = pd.DataFrame(
            data,
            columns=[
                'seq',
                'history_seq',
                'len_seq',
                'next'
            ]
        )

        df.to_pickle(file_path)

    save_to_file(
        train_data,
        train_file
    )

    save_to_file(
        val_data,
        val_file
    )

    save_to_file(
        test_data,
        test_file
    )


# ================================================================
# Movie
# ================================================================

input_csv = "./dataset/Amazon/ratings_Movies_and_TV.csv"

train_df = "./data/movie/train.df"

val_df = "./data/movie/val.df"

test_df = "./data/movie/test.df"


process_csv(
    input_csv,
    train_df,
    val_df,
    test_df,
    target_length=8,
    history_length=16
)