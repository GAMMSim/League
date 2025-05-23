from typeguard import typechecked
import hashlib
import json
import time
import os

try:
    from core import *
except ImportError:
    from lib.core.core import *


@typechecked
class TimeLogger:
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None, path: str = ""):
        """
        Initialize the TimeLogger with a name, optional global metadata, and a file path.

        Args:
            name (str): The name of the logger.
            metadata (Dict[str, Any], optional): Global parameters for the log file.
            path (str, optional): The directory path to read/write files.
        """
        self.name: str = name
        self.records: List[Dict[str, Any]] = []  # Each record is a dict with 'time' (float) and custom columns.
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.path: str = path

    def log_data(self, data: dict, time: float) -> None:
        """
        Log custom data with a provided timestamp.

        Args:
            data (dict): Dictionary containing custom data.
            time (float): The timestamp for the log entry.
        """
        record = {"time": time, **data}
        self.records.append(record)

    def get_records(self) -> List[Dict[str, Any]]:
        """
        Return all logged records.
        """
        return self.records

    def set_records(self, records: List[Dict[str, Any]]) -> None:
        """
        Replace the current records with a new list.

        Args:
            records (List[Dict[str, Any]]): New list of records.
        """
        self.records = records

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return the global metadata for the log file.
        """
        return self.metadata

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set the global metadata for the log file.

        Args:
            metadata (Dict[str, Any]): A dictionary containing global parameters.
        """
        self.metadata = metadata

    def write_to_file(self, filename: str = None, force: bool = False) -> None:
        """
        Write the current metadata and log records to a file in JSON format.
        If filename is not specified, generate one based on metadata and current time.
        If the generated or provided filename already exists in the given path and force is False,
        raise an error and abort.

        Args:
            filename (str, optional): The name of the file where the log data will be saved.
            force (bool): If True, overwrite an existing file.
        """
        if filename is None:
            meta_str = json.dumps(self.metadata, sort_keys=True)
            current_time = str(time.time())
            hash_input = (meta_str + current_time).encode("utf-8")
            filename = "result" + hashlib.sha256(hash_input).hexdigest() + ".json"

        full_path = os.path.join(self.path, filename)
        if os.path.exists(full_path) and not force:
            raise FileExistsError(f"File {full_path} already exists. Aborting write.")

        data_to_write = {"metadata": self.metadata, "records": self.records}
        with open(full_path, "w") as f:
            json.dump(data_to_write, f, indent=4)

    def read_from_file(self, filename: str) -> None:
        """
        Read metadata and log records from a JSON file located in the given path and update the current state.

        Args:
            filename (str): The name of the file to read from.
        """
        full_path = os.path.join(self.path, filename)
        with open(full_path, "r") as f:
            data_loaded = json.load(f)
            self.metadata = data_loaded.get("metadata", {})
            self.records = data_loaded.get("records", [])

    def extract_by_time(self, time_min: float, time_max: float) -> List[Dict[str, Any]]:
        """
        Extract records with time values within a given range (inclusive).

        Args:
            time_min (float): The minimum time value.
            time_max (float): The maximum time value.

        Returns:
            List[Dict[str, Any]]: A list of records with 'time' between time_min and time_max.
        """
        return [record for record in self.records if time_min <= record["time"] <= time_max]

    def extract_by_keys(self, keys: List[str]) -> List[Dict[str, Any]]:
        """
        Extract specific columns from each record.

        Args:
            keys (List[str]): The list of keys (columns) to extract.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing only the specified keys,
                                  if they exist in each record.
        """
        extracted = []
        for record in self.records:
            # Always include time if not explicitly omitted
            new_record = {"time": record["time"]} if "time" not in keys else {}
            for key in keys:
                if key in record:
                    new_record[key] = record[key]
            extracted.append(new_record)
        return extracted

    def finalize(self, **summary_data: Any) -> None:
        """
        Append a final summary record containing overall game statistics or any extra data.

        Accepts arbitrary keyword arguments which are added to a summary record with a flag 'final_summary'.

        Args:
            **summary_data: Arbitrary keyword arguments representing summary data.
        """
        summary = {"summary": True}
        summary.update(summary_data)
        self.records.append(summary)

    def extract_metadata(self) -> Dict[str, Any]:
        """
        Extract and return the global metadata for this logger.

        Returns:
            Dict[str, Any]: The metadata dictionary.
        """
        return self.metadata

    def extract_summary(self) -> List[Dict[str, Any]]:
        """
        Extract and return all final summary records from the log.

        Returns:
            List[Dict[str, Any]]: A list of summary records (records with 'final_summary' set to True).
        """
        return [record for record in self.records if record.get("summary", False)]


# Example usage:
if __name__ == "__main__":
    # Create an instance of TimeLogger with some global metadata
    logger = TimeLogger("MyTimeLogger", metadata={"version": "1.0", "description": "Test log file"})

    # Log some data with custom columns and a provided time (as float)
    logger.log_data({"temperature": 23.5, "humidity": 60}, time=2.0)
    logger.log_data({"temperature": 24.0, "humidity": 58}, time=1.0)

    # Retrieve and print global metadata and all logged records
    print("Global Metadata:")
    print(logger.get_metadata())

    print("\nAll Records:")
    for record in logger.get_records():
        print(record)

    # Extract records in a time range
    print("\nRecords between time 1.0 and 1.0:")
    for record in logger.extract_by_time(1.0, 1.0):
        print(record)

    # Extract only the temperature column (time is always included unless omitted)
    print("\nExtracted Keys (['temperature']):")
    for record in logger.extract_by_keys(["temperature"]):
        print(record)

    # Write metadata and records to a file
    logger.write_to_file("log_records.json")

    # To demonstrate reading, clear current records and metadata, then read from file
    logger.set_records([])
    logger.set_metadata({})
    logger.read_from_file("log_records.json")

    print("\nAfter reading from file:")
    print("Global Metadata:")
    print(logger.get_metadata())
    print("All Records:")
    for record in logger.get_records():
        print(record)
