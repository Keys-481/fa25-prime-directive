import os
import csv
import json
import re
from pathlib import Path

# ========== DEBUG UTIL ==========
def debug(msg):
    print(f"[DEBUG] {msg}")

# ========== CLEANING FUNCTION ==========
def clean_value(value):
    """Remove std_msgs wrappers and extra formatting noise."""
    if isinstance(value, str):
        value = re.sub(r"std_msgs\.msg\.\w+\(data='(.*)'\)", r"\1", value)
        value = value.strip().replace("\n", "")
    return value

# ========== PARSERS ==========
def parse_csv(file_path):
    debug(f"Parsing CSV file: {file_path}")
    rows, headers = [], []
    with open(file_path, encoding='utf-8') as f:
        sample = f.read(1024)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.reader(f, delimiter=dialect.delimiter)
        headers = next(reader)
        headers = [h.strip() for h in headers]
        for row in reader:
            rows.append([clean_value(c) for c in row])
    debug(f"CSV headers: {headers}")
    debug(f"CSV rows extracted ({len(rows)}): {rows}")
    return headers, rows

def parse_json(file_path):
    debug(f"Parsing JSON file: {file_path}")
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    # special case for all_topics.json
    if os.path.basename(file_path) == "all_topics.json":
        return parse_all_topics_json(data)

    headers, rows = [], []
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            headers = list(data[0].keys())
            for d in data:
                rows.append([clean_value(d.get(h, "")) for h in headers])
        else:
            headers = ["Value"]
            rows = [[clean_value(v)] for v in data]
    elif isinstance(data, dict):
        headers = list(data.keys())
        rows = [[clean_value(data[h]) for h in headers]]
    else:
        headers = ["Value"]
        rows = [[clean_value(data)]]

    debug(f"JSON headers: {headers}")
    debug(f"JSON rows extracted ({len(rows)}): {rows}")
    return headers, rows

def parse_all_topics_json(data):
    """Properly flatten ROS2-style all_topics.json."""
    debug("Parsing all_topics.json structure")

    headers = ["topic", "timestamp", "data"]
    rows = []

    if not data:
        debug("Empty JSON data in all_topics.json")
        return headers, rows

    # Expecting a list containing one dict of topics
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for topic, messages in item.items():
                    if isinstance(messages, list):
                        for msg in messages:
                            timestamp = msg.get("timestamp", "")
                            message_data = clean_value(msg.get("data", ""))
                            rows.append([topic, timestamp, message_data])
    elif isinstance(data, dict):
        for topic, messages in data.items():
            if isinstance(messages, list):
                for msg in messages:
                    timestamp = msg.get("timestamp", "")
                    message_data = clean_value(msg.get("data", ""))
                    rows.append([topic, timestamp, message_data])

    debug(f"Flattened all_topics.json rows ({len(rows)}): {rows}")
    return headers, rows

# ========== HTML GENERATION ==========
def generate_table_html(file_name, headers, rows):
    debug(f"Generating HTML table for: {file_name}")
    html = f"<h2>{file_name}</h2>\n<table border='1'>\n<tr>"
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr>\n"

    for row in rows:
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>\n"
    html += "</table>\n"
    debug(f"[TABLE BUILT] {file_name} with {len(rows)} rows")
    return html

def generate_folder_html(folder_name, folder_path, output_dir):
    debug(f"Generating HTML page for folder: {folder_name}")
    html_content = f"<html><head><title>{folder_name}</title></head><body>\n"
    html_content += f"<h1>Folder: {folder_name}</h1>\n"

    for file in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            try:
                if ext == ".csv":
                    headers, rows = parse_csv(file_path)
                    html_content += generate_table_html(file, headers, rows)
                elif ext == ".json":
                    headers, rows = parse_json(file_path)
                    html_content += generate_table_html(file, headers, rows)
            except Exception as e:
                debug(f"[ERROR] Failed to parse {file}: {e}")

    html_content += "</body></html>"
    output_file = os.path.join(output_dir, f"{folder_name}.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    debug(f"[DONE] Folder HTML generated: {output_file}")
    return output_file

def generate_index_html(folder_names, output_dir):
    debug("Generating index.html for all folders")
    html_content = "<html><head><title>Index</title></head><body>\n<h1>Folders</h1>\n<ul>\n"
    for folder in folder_names:
        html_content += f"<li><a href='{folder}.html'>{folder}</a></li>\n"
    html_content += "</ul>\n</body></html>"
    output_file = os.path.join(output_dir, "index.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    debug(f"[DONE] index.html generated: {output_file}")
    return output_file

# ========== MAIN ==========
def main():
    base_dir = os.getcwd()
    debug(f"Base directory: {base_dir}")

    csv_json_dir = os.path.join(base_dir, "CSV_JSON_FILES")
    debug(f"CSV_JSON_FILES directory: {csv_json_dir}")

    html_output_dir = os.path.join(base_dir, "html_web_page")
    Path(html_output_dir).mkdir(parents=True, exist_ok=True)
    debug(f"HTML output folder ensured at {html_output_dir}")

    folder_names = []
    if os.path.exists(csv_json_dir):
        for folder in sorted(os.listdir(csv_json_dir)):
            folder_path = os.path.join(csv_json_dir, folder)
            if os.path.isdir(folder_path):
                folder_names.append(folder)
                generate_folder_html(folder, folder_path, html_output_dir)

    debug(f"Found folders: {folder_names}")
    generate_index_html(folder_names, html_output_dir)

if __name__ == "__main__":
    main()
