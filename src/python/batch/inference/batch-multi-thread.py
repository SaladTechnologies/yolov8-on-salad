from azure.storage.blob import ContainerClient
from azure.storage.queue import QueueClient
import pandas as pd
import cv2
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube
import pafy
import datetime
import json
import csv
import os
import io
import argparse
import logging.config
import uuid
import base64
import torch
import threading
import time


def dequeue_message(queue_client, queue_message):
    queue_client.delete_message(queue_message.id, queue_message.pop_receipt)


def connect_to_storage(storage_type: str, name: str, connection_str: str):
    """_summary_

    Args:
        storage_type (str): an azure storage type
        name (str): name of specific storage item (container name or queue name)

    Raises:
        ValueError: Error raised when storage type is not an azure storage option

    Returns:
        ContainerClient | QueueClient: Azure object used to interact with Azure Storage
    """

    if storage_type.lower() == "container":
        client = ContainerClient
    elif storage_type.lower() == "queue":
        client = QueueClient
    else:
        client = None

    if client:
        azure_client = client.from_connection_string(connection_str, name)
        return azure_client
    else:
        raise ValueError("Storage type should be container or queue")


def azure_initiate(
    input_blob: str,
    output_blob: str,
    fail_blob: str,
    storage_queue: str,
    poison_queue: str,
    storage_connection_string: str,
):
    request = connect_to_storage("container", input_blob, storage_connection_string)
    result = connect_to_storage("container", output_blob, storage_connection_string)
    fail = connect_to_storage("container", fail_blob, storage_connection_string)
    queue = connect_to_storage("queue", storage_queue, storage_connection_string)
    poison_queue = connect_to_storage("queue", poison_queue, storage_connection_string)

    return request, result, fail, queue, poison_queue


def retrieve_file(container_client, file_name):
    return container_client.get_blob_client(file_name)


def decode_queue_message(b64_message):
    # message content coming from queue is base64 encoded. We decode it and read as json

    decoded_message = base64.b64decode(b64_message)
    json_message = json.loads(decoded_message)
    return json_message


def retrieve_blob_details(queue_message, input_blob_name):
    decoded_message = decode_queue_message(queue_message.content)
    blob_url = decoded_message["data"]["url"]

    blob_name = blob_url.rsplit("/", 1)[-1]
    file_ext = blob_name.split(".")[1]
    return blob_name, file_ext, decoded_message


def calculate_percentage(bbox, original_shape):
    bbox_area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
    original_shape_area = original_shape[0] * original_shape[1]
    percentage = (bbox_area / original_shape_area) * 100
    return percentage


def summary(df, filename, result_blob, processing_time=None):
    if (
        "track_id" in df.columns
        and df["track_id"].notna().any()
        and df["track_id"].ne(0).any()
    ):
        df_filtered = df[(df["track_id"] != 0) & (df["track_id"].notna())].copy()
        # Group by 'track_id' and calculate duration, most frequent class and
        # corresponding name for each group
        # Group by track_id and calculate average box_percentage, min and max timestamp
        summary_df = (
            df_filtered.groupby("track_id")
            .agg(
                average_box_percentage=("box_percentage", "mean"),
                min_timestamp=("timestamp", "min"),
                max_timestamp=("timestamp", "max"),
                most_common_class=(
                    "name",
                    lambda x: x.value_counts().index[0],
                ),  # Most common class per track_id
            )
            .reset_index()
        )
        # Calculate duration
        summary_df["duration"] = (
            summary_df["max_timestamp"] - summary_df["min_timestamp"]
        )

        # Convert the DataFrame to a string
        output_string = "\n".join(
            f"{row['most_common_class']} with id {row['track_id']} was present in the video for {row['duration']} from {row['min_timestamp']} to {row['max_timestamp']} and was taking  {row['average_box_percentage']:.2f}% of the screen"
            for _, row in summary_df.iterrows()
        )
        # if processing_time is not None, append it to output_string
        if processing_time is not None:
            output_string += f"\nProcessing time: {processing_time}\n"
    else:
        output_string = "No objects were detected in the video"

    results_txt_file_name = f"{filename}.txt"
    results_blob_client_txt = result_blob.get_blob_client(results_txt_file_name)
    results_blob_client_txt.upload_blob(output_string, overwrite=True)


def save_df(df, filename, result_blob):
    results_csv_file_name = f"{filename}.csv"
    results_blob_client = result_blob.get_blob_client(results_csv_file_name)
    csv_stream = io.StringIO()
    df.to_csv(csv_stream, index=False)
    # Convert the CSV data to bytes
    csv_bytes = csv_stream.getvalue().encode("utf-8")
    results_blob_client.upload_blob(csv_bytes, overwrite=True)


# Function that will be targt for the thread
def run_tracker_in_thread(link, live, model, result_blob, save_every_mins):
    """
    This function is designed to run a yutube or webcam stream
    concurrently with the YOLOv8 model, utilizing threading.

    - link: The path to video or the webcam/external
    camera source.
    - model: The file path to the YOLOv8 model.
    - file_index: An argument to specify the count of the
    file being processed.
    """
    # Process a youtube link:
    if live == "not-live":
        cap = cap_from_youtube(link, "720p")
    # Process a streaming video
    if live and ("rtsp" in link or "rtmp" in link or "tcp" in link):
        cap = cv2.VideoCapture(link)
    # Process a streaming video from youtube
    elif live:
        video = pafy.new(link)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)

    # we will store all the results as a list of dictionaries
    all_results = []
    starttime = datetime.datetime.now()
    last_save_time = starttime
    filename = (
        link.split("=")[-1] + "_" + str(starttime.time().strftime("%Y-%m-%d-%H-%M-%S"))
    )
    # we will store all the results as a list of dictionaries
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            results = model.track(frame, persist=True)
            timestamp = datetime.datetime.now()
            # save every box with label
            for box in json.loads(results[0].tojson()):
                box["input"] = link
                box["timestamp"] = timestamp
                box["date"] = timestamp.strftime("%Y-%m-%d")
                box["time"] = timestamp.time().strftime("%H:%M:%S")
                box["origin_shape"] = results[0].orig_shape
                box["box_percentage"] = calculate_percentage(
                    box["box"], results[0].orig_shape
                )
                box["full_process_speed"] = sum(results[0].speed.values())
                all_results.append(box)

                # Get the current time
                current_time = datetime.datetime.now()
                # Check if 30 minutes have passed since the last save
                if (current_time - last_save_time).total_seconds() >= 30 * 60:
                    df = pd.DataFrame(all_results)
                    save_df(df, filename, result_blob)
                    summary(df, filename, result_blob)
                    last_save_time = current_time

        # Break the loop if the process should not continue
        else:
            finishtime = datetime.datetime.now()
            processing_time = finishtime - starttime
            df = pd.DataFrame(all_results)
            save_df(df, filename, result_blob)
            summary(df, filename, result_blob, processing_time)
            break


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--storage_connection_string",
            default=os.environ.get("STORAGE_CONNECTION_STRING"),
            type=str,
        )
        parser.add_argument(
            "--input_container_name",
            default=os.environ.get("INPUT_CONTAINER_NAME", "requests"),
            type=str,
        )
        parser.add_argument(
            "--output_container_name",
            default=os.environ.get("OUTPUT_CONTAINER_NAME", "yolo-results"),
            type=str,
        )
        parser.add_argument(
            "--fail_container_name",
            default=os.environ.get("FAIL_CONTAINER_NAME", "failedrequests"),
            type=str,
        )
        parser.add_argument(
            "--queue_name",
            default=os.environ.get("INPUT_QUEUE_NAME", "requestsqueue"),
            type=str,
        )
        parser.add_argument(
            "--poison_queue_name",
            default=os.environ.get("POISON_QUEUE_NAME", "poison-requestsqueue"),
            type=str,
        ),
        parser.add_argument(
            "--save_every_mins",
            default=os.environ.get("SAVE_EVERY_MINS", 30),
            type=int,
        ),
        parser.add_argument(
            "--queue_check_timer",
            default=os.environ.get("QUEUE_CHECK_TIMER", 60),
            type=int,
        )
        args = parser.parse_args()

        # Check for CUDA device and set it
        device = "0" if torch.cuda.is_available() else "cpu"
        if device == "0":
            torch.cuda.set_device(0)

        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")

        # authentiacate in azure
        (
            request_blob_client,
            result_blob_client,
            fail_blob_client,
            request_queue_client,
            poison_queue_client,
        ) = azure_initiate(
            args.input_container_name,
            args.output_container_name,
            args.fail_container_name,
            args.queue_name,
            args.poison_queue_name,
            args.storage_connection_string,
        )
        logging.info("Connected to azure containers and queues")

        while True:
            logging.info("Checking for messages in queue")
            queue_length = (
                request_queue_client.get_queue_properties().approximate_message_count
            )
            if queue_length > 0:
                queue_messages = request_queue_client.receive_messages(
                    visibility_timeout=int("1")
                )

                for queue_message in queue_messages:
                    message_correlation_id = str(uuid.uuid4())
                    (
                        analysis_file_name,
                        file_ext,
                        decoded_message,
                    ) = retrieve_blob_details(queue_message, args.input_container_name)
                    analysis_file = retrieve_file(
                        request_blob_client, analysis_file_name
                    )
                    # send message to azure insights:
                    analysis_request = io.BytesIO(
                        analysis_file.download_blob().readall()
                    )
                    decoded_message = analysis_request.getvalue().decode("utf-8")
                    links = json.loads(decoded_message)
                    dequeue_message(request_queue_client, queue_message)
                    tracker_treads = []
                    for link in links:
                        url = link["url"]
                        status = link["status"]
                        tracker_tread = threading.Thread(
                            target=run_tracker_in_thread,
                            args=(
                                url,
                                status,
                                model,
                                result_blob_client,
                                args.save_every_mins,
                            ),
                            daemon=True,
                        )
                        tracker_treads.append(tracker_tread)
                    for tracker_tread in tracker_treads:
                        tracker_tread.start()

                    for tracker_tread in tracker_treads:
                        tracker_tread.join()
            else:
                time.sleep(args.queue_check_timer)

    except Exception as exc:
        logging.error(exc)
        raise
    finally:
        print("done")


if __name__ == "__main__":
    main()
