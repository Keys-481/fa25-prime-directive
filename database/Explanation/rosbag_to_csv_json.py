#!/usr/bin/env python3
import os
import sys
import rosbag2_py
import pandas as pd
import json
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from datetime import datetime

def convert_rosbag(bag_folder):
    if not os.path.exists(bag_folder):
        print(f"Path not found: {bag_folder}")
        sys.exit(1)

    # Automatically find .mcap file
    mcap_file = None
    for file in os.listdir(bag_folder):
        if file.endswith(".mcap"):
            mcap_file = file
            break

    if not mcap_file:
        print(f"No .mcap file found in {bag_folder}")
        sys.exit(1)

    print(f"Found bag file: {mcap_file}")

    # Base output folder
    base_output_folder = os.path.join(os.path.expanduser("~/rosbag_test"), "CSV_JSON_FILES")
    os.makedirs(base_output_folder, exist_ok=True)

    # Subfolder named after the bag folder
    bag_name = os.path.basename(os.path.normpath(bag_folder))
    output_folder = os.path.join(base_output_folder, bag_name)
    os.makedirs(output_folder, exist_ok=True)

    # Open rosbag
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_folder, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics}

    # Data storage
    data_per_topic = {topic.name: [] for topic in topics}

    print("Reading messages...")
    while reader.has_next():
        topic_name, data_bytes, t = reader.read_next()
        try:
            msg_type = get_message(topic_types[topic_name])
            msg = deserialize_message(data_bytes, msg_type)
            timestamp = t / 1e9
            entry = {
                "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"),
                "data": str(msg)
            }
            data_per_topic[topic_name].append(entry)
        except Exception as e:
            print(f"Could not parse message on topic {topic_name}: {e}")

    # Export each topic to CSV and JSON
    combined_json = {}
    for topic, messages in data_per_topic.items():
        if not messages:
            print(f"No messages for topic {topic}")
            continue

        safe_topic = topic.replace("/", "_").strip("_")

        # CSV
        csv_path = os.path.join(output_folder, f"{safe_topic}.csv")
        pd.DataFrame(messages).to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")

        # JSON
        json_path = os.path.join(output_folder, f"{safe_topic}.json")
        with open(json_path, "w") as f:
            json.dump(messages, f, indent=2)
        print(f"JSON saved: {json_path}")

        # Add to combined JSON
        combined_json[topic] = messages

    # Combined JSON
    combined_path = os.path.join(output_folder, "all_topics.json")
    with open(combined_path, "w") as f:
        json.dump(combined_json, f, indent=2)
    print(f"Combined JSON saved: {combined_path}")
    print(f"All CSV and JSON files for this bag are in: {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python3 rosbag_to_csv_json.py <rosbag_folder_path>")
        sys.exit(1)
    bag_folder = sys.argv[1]
    convert_rosbag(bag_folder)