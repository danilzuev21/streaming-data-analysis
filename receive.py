import os
import time
from json import loads

import numpy as np
import pika
import sys
import pandas as pd
import torch

import torch.nn.functional as F
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from torch import Tensor, nn, stack
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

input_cols = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]
output_cols = ['quality']

input_size = len(input_cols)
output_size = len(output_cols)


class WineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)  # fill this (hint: use input_size & output_size defined above)
        # model initialized with random weight

    def forward(self, xb):
        out = self.linear(xb)  # batch wise forwarding
        return out


def string_to_pd(df_string: str) -> pd.DataFrame:
    df_dict = loads(df_string)
    return pd.DataFrame(df_dict)


def dataframe_to_nd_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Make a copy of the original dataframe
    df1 = df.copy(deep=True)
    # Extract input & outputs as numpy arrays
    inputs_array = df1[input_cols].to_numpy()
    targets_array = df1[output_cols].to_numpy()
    return inputs_array, targets_array


def array_to_tensor(array: np.ndarray) -> Tensor:
    return Tensor(array)


def get_tensor_dataset(inputs: Tensor, outputs: Tensor) -> TensorDataset:
    return TensorDataset(inputs, outputs)


def get_data_loader(dataset: TensorDataset) -> DataLoader:
    return DataLoader(dataset)


def prepare_data(df: pd.DataFrame) -> DataLoader:
    inputs_array, targets_array = dataframe_to_nd_arrays(df)
    inputs_tensor = array_to_tensor(inputs_array)
    targets_tensor = array_to_tensor(targets_array)
    dataset = get_tensor_dataset(inputs_tensor, targets_tensor)
    return get_data_loader(dataset)


def train_on_batch(model: nn.Module,
                   optimizer: Adam,
                   criterion: nn.L1Loss,
                   data_loader: DataLoader,
                   device,
                   batch_num: int) -> tuple[float, float]:
    num_epochs = 3
    for epoch_num in range(num_epochs):
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track train loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
        train_accuracy = correct_train / total_train
        train_loss = train_loss / total_train
    # print(f"Batch № {batch_num} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f}")
    return train_loss, train_accuracy


def get_model() -> nn.Module:
    return WineModel()


def get_optimizer(model: nn.Module) -> Adam:
    return Adam(model.parameters())


def get_loss():
    return nn.L1Loss()


def init_connection() -> tuple[BlockingConnection, BlockingChannel, int]:
    connection = BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    q = channel.queue_declare(queue='wine_quality')
    print(q.method.message_count)
    return connection, channel, q.method.message_count


def close_connection(connection: BlockingConnection):
    connection.close()


class MessageProcessor:
    def __init__(self, connection: BlockingConnection, channel: BlockingChannel):
        self.connection = connection
        self.channel = channel

        self.model = get_model()
        self.optimizer = get_optimizer(self.model)
        self.loss_fn = get_loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.batch_num = 0
        self.data = []
        self.df_data = pd.DataFrame(columns=[
            'fixed acidity',
            'volatile acidity',
            'citric acid',
            'residual sugar',
            'chlorides',
            'free sulfur dioxide',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol',
            'quality'
        ])
        self.train_accuracies = []
        self.train_losses = []

        self.batch_size = 0
        self.process_time = 0
        self.accuracy = 0
        self.loss = 100
        self.result_cols = ['batch_size', 'process_time', 'accuracy', 'loss']

        self.result_file = 'result.csv'
        self.result_df: pd.DataFrame = pd.DataFrame()

    def run_train(self, df: pd.DataFrame):
        start_time = time.process_time()
        data_loader = prepare_data(df)
        train_accuracy, train_loss = train_on_batch(
            self.model, self.optimizer, self.loss_fn, data_loader, self.device, self.batch_num
        )
        batch_process_time = time.process_time() - start_time
        self.process_time += batch_process_time
        self.train_accuracies.append(train_accuracy)
        self.train_losses.append(train_loss)

    def end_consume(self):
        self.channel.queue_delete('wine_quality')
        self.channel.stop_consuming()
        close_connection(self.connection)

    def append_df(self, batch_df: pd.DataFrame):
        if self.df_data.shape[0] == 0:
            self.df_data = batch_df
        else:
            self.df_data = pd.concat([self.df_data, batch_df], ignore_index=True)

    def evaluate_model(self):
        inputs_array, targets_array = dataframe_to_nd_arrays(self.df_data)
        inputs_tensor = array_to_tensor(inputs_array).to(self.device)
        targets_tensor = array_to_tensor(targets_array).to(self.device)
        dataset = get_tensor_dataset(inputs_tensor, targets_tensor)
        data_loader = DataLoader(dataset, batch_size=inputs_array.shape[0])
        for inputs, targets in data_loader:
            inputs.to(self.device)
            targets.to(self.device)
            outputs = self.model(inputs)
            predicted = torch.round(outputs)
            correct_predictions = (predicted == targets).sum().item()

            self.accuracy = correct_predictions / targets_array.shape[0]
            self.loss = self.loss_fn(targets, outputs).item()
            break

    def init_result_df(self):
        if os.path.isfile(self.result_file):
            self.result_df = pd.read_csv(self.result_file)
        else:
            self.result_df = pd.DataFrame(columns=[self.result_cols])

    def save_result(self):
        self.evaluate_model()
        self.init_result_df()
        print(self.batch_size)
        print(self.process_time)
        print(self.accuracy)
        print(self.loss)
        added_df = pd.DataFrame([{
            'batch_size': self.batch_size,
            'process_time': self.process_time,
            'accuracy': self.accuracy,
            'loss': self.loss
        }])
        if self.result_df.shape[0] == 0:
            self.result_df = added_df
        else:
            self.result_df = pd.concat([self.result_df, added_df], ignore_index=True)
        print(self.result_df.tail().to_string())
        self.result_df.to_csv(self.result_file, index=False)

    def end_processing(self):
        self.end_consume()
        self.save_result()

    def process_message(self, body: bytes):
        decoded_string = body.decode('utf-8')
        if decoded_string == 'End':
            self.end_processing()
            return
        self.process_batch(decoded_string)

    def process_batch(self, json_string: str):
        self.batch_num += 1
        batch_df = string_to_pd(json_string)
        if self.batch_size == 0:
            self.batch_size = batch_df.shape[0]
        self.append_df(batch_df)
        self.run_train(batch_df)


def main():
    connection, channel, messages_count = init_connection()
    message_processor = MessageProcessor(connection, channel)
    for i in range(messages_count):
        method_frame, properties, body = channel.basic_get('wine_quality')
        # print(f" [x] Process started")
        message_processor.process_message(body)
        # print(" [x] Process ended")

    # def callback(ch, method, properties, body):
    #     pass
    #     # print(f" [x] Process started")
    #     # message_processor.process_message(body)
    #     # print(" [x] Process ended")

    # channel.basic_consume(queue='wine_quality', on_message_callback=callback, auto_ack=True)
    #
    # print(' [*] Waiting for messages. To exit press CTRL+C')
    # channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
