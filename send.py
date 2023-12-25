from json import loads
from typing import Iterator

import pika
import pandas as pd
from pika.adapters.blocking_connection import BlockingChannel


def load_data() -> pd.DataFrame:
    return pd.read_csv("WineQT.csv")


def iterate_data(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    df_len = df.shape[0]
    for batch_begin_idx in range(0, df_len, batch_size):
        batch_end_idx = batch_begin_idx + batch_size if batch_begin_idx + batch_size < df_len else df_len
        batch_end_idx = batch_end_idx - 1
        yield df.loc[batch_begin_idx:batch_end_idx]


def send_batch_data(df_batch: pd.DataFrame, channel: BlockingChannel):
    list_to_send = df_batch.to_json(orient="records")
    channel.basic_publish(exchange='', routing_key='wine_quality', body=list_to_send.encode('utf-8'))


def mark_as_end(channel: BlockingChannel):
    channel.basic_publish(exchange='', routing_key='wine_quality', body='End')


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    q = channel.queue_declare(queue='wine_quality')
    df = load_data()
    batch_size = 510
    for df_batch in iterate_data(df, batch_size):
        send_batch_data(df_batch, channel)
    mark_as_end(channel)
    connection.close()


main()
