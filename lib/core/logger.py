from typeguard import typechecked
import hashlib
import json
import time
import os
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

try:
    import cbor2

    CBOR_AVAILABLE = True
except ImportError:
    CBOR_AVAILABLE = False

try:
    from console import *
except ImportError:
    from lib.core.console import *


class LoggerError(Exception):
    """Custom exception for Logger-related errors."""

    pass


class ValidationError(LoggerError):
    """Exception raised when data validation fails."""

    pass


class FileFormatError(LoggerError):
    """Exception raised when file format is not supported or corrupted."""

    pass


@typechecked
class Logger:
    """
    Enhanced logging system supporting both JSON and CBOR formats with validation and performance optimizations.
    """

    # Supported file formats
    SUPPORTED_FORMATS = {".json": "json", ".cbor": "cbor"}

    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None, path: str = "", validate_schema: bool = True, filename_pattern: str = "{name}_{hash}_{timestamp}"):
        """
        Initialize Logger with enhanced features.

        Args:
            name (str): The name of the logger.
            metadata (Dict[str, Any], optional): Global parameters for the log file.
            path (str, optional): The directory path to read/write files.
            validate_schema (bool): Enable schema validation for consistency.
            filename_pattern (str): Pattern for auto-generated filenames.
        """
        self.name: str = name
        self._records: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.path: str = path
        self.validate_schema: bool = validate_schema
        self.filename_pattern: str = filename_pattern

        # Internal state for validation and performance
        self._schema_keys: Optional[set] = None
        self._record_count: int = 0
        self._time_range: Optional[tuple] = None

        # Check CBOR availability on first use
        if not CBOR_AVAILABLE:
            warning("CBOR support not available. Install cbor2 package for CBOR functionality.")

    def _validate_record(self, record: Dict[str, Any]) -> None:
        """Validate record structure and consistency."""
        if not self.validate_schema:
            return

        if not isinstance(record, dict):
            raise ValidationError(f"Record must be a dictionary, got {type(record)}")

        if "time" not in record:
            raise ValidationError("Record must contain 'time' field")

        if not isinstance(record["time"], int):
            raise ValidationError(f"Time field must be an integer, got {type(record['time'])}")

        # Schema consistency check
        if self._schema_keys is None:
            self._schema_keys = set(record.keys())
        else:
            current_keys = set(record.keys())
            if current_keys != self._schema_keys:
                missing = self._schema_keys - current_keys
                extra = current_keys - self._schema_keys
                warnings = []
                if missing:
                    warnings.append(f"Missing keys: {missing}")
                if extra:
                    warnings.append(f"Extra keys: {extra}")
                warning(f"Schema inconsistency detected: {', '.join(warnings)}")

    def _update_stats(self, record: Dict[str, Any]) -> None:
        """Update internal statistics for performance optimization."""
        self._record_count += 1

        # Update time range
        record_time = record["time"]
        if self._time_range is None:
            self._time_range = (record_time, record_time)
        else:
            min_time, max_time = self._time_range
            self._time_range = (min(min_time, record_time), max(max_time, record_time))

    def _detect_format(self, filename: str) -> str:
        """Detect file format from extension."""
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise FileFormatError(f"Unsupported file format: {suffix}. Supported formats: {list(self.SUPPORTED_FORMATS.keys())}")
        return self.SUPPORTED_FORMATS[suffix]

    def _generate_filename(self, format_ext: str = ".json") -> str:
        """Generate filename using the configured pattern."""
        meta_str = json.dumps(self.metadata, sort_keys=True)
        current_time = str(time.time())
        hash_input = (meta_str + current_time).encode("utf-8")
        hash_value = hashlib.sha256(hash_input).hexdigest()[:16]  # Shorter hash
        timestamp = str(int(time.time()))

        filename = self.filename_pattern.format(name=self.name, hash=hash_value, timestamp=timestamp, metadata_hash=hashlib.sha256(meta_str.encode()).hexdigest()[:8])
        return filename + format_ext

    def _write_data(self, data: Dict[str, Any], filepath: str, file_format: str) -> None:
        """Write data in the specified format with error handling."""
        try:
            if file_format == "json":
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=4)
            elif file_format == "cbor":
                if not CBOR_AVAILABLE:
                    raise FileFormatError("CBOR format not available. Install cbor2 package.")
                with open(filepath, "wb") as f:
                    cbor2.dump(data, f)
            else:
                raise FileFormatError(f"Unsupported format: {file_format}")
        except (IOError, OSError) as e:
            raise LoggerError(f"Failed to write file {filepath}: {str(e)}")
        except Exception as e:
            raise LoggerError(f"Unexpected error writing {filepath}: {str(e)}")

    def _read_data(self, filepath: str, file_format: str) -> Dict[str, Any]:
        """Read data in the specified format with error handling."""
        try:
            if file_format == "json":
                with open(filepath, "r") as f:
                    return json.load(f)
            elif file_format == "cbor":
                if not CBOR_AVAILABLE:
                    raise FileFormatError("CBOR format not available. Install cbor2 package.")
                with open(filepath, "rb") as f:
                    return cbor2.load(f)
            else:
                raise FileFormatError(f"Unsupported format: {file_format}")
        except (IOError, OSError) as e:
            raise LoggerError(f"Failed to read file {filepath}: {str(e)}")
        except json.JSONDecodeError as e:
            raise FileFormatError(f"Invalid JSON in file {filepath}: {str(e)}")
        except Exception as e:
            raise LoggerError(f"Unexpected error reading {filepath}: {str(e)}")

    def log_data(self, data: dict, time_value: int) -> None:
        """
        Log custom data with timestamp.

        Args:
            data (dict): Dictionary containing custom data.
            time_value (int): The timestamp (integer) for the log entry.
        """
        if not isinstance(time_value, int):
            raise ValidationError(f"Time value must be an integer, got {type(time_value)}")

        record = {"time": time_value, **data}

        try:
            self._validate_record(record)
            self._records.append(record)
            self._update_stats(record)
        except ValidationError as e:
            error(f"Validation failed for record: {str(e)}")
            raise

    def get_records(self) -> List[Dict[str, Any]]:
        """Return all logged records."""
        return self._records.copy()  # Return copy for safety

    def set_records(self, records: List[Dict[str, Any]]) -> None:
        """
        Replace the current records with a new list.

        Args:
            records (List[Dict[str, Any]]): New list of records.
        """
        if self.validate_schema:
            for i, record in enumerate(records):
                try:
                    self._validate_record(record)
                except ValidationError as e:
                    raise ValidationError(f"Invalid record at index {i}: {str(e)}")

        self._records = records
        self._record_count = len(records)

        # Recalculate time range
        self._time_range = None
        if records:
            times = [r["time"] for r in records]
            self._time_range = (min(times), max(times))

    def get_metadata(self) -> Dict[str, Any]:
        """Return the global metadata for the log file."""
        return self.metadata.copy()  # Return copy for safety

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set the global metadata for the log file.

        Args:
            metadata (Dict[str, Any]): A dictionary containing global parameters.
        """
        self.metadata = metadata

    def write_to_file(self, filename: Optional[str] = None, force: bool = False, format: str = "auto") -> str:
        """
        Write the current metadata and log records to a file.

        Args:
            filename (str, optional): The name of the file. If None, auto-generate.
            force (bool): If True, overwrite an existing file.
            format (str): File format ('json', 'cbor', or 'auto' to detect from extension).

        Returns:
            str: The actual filename used.
        """
        if filename is None:
            filename = self._generate_filename(".json" if format == "json" else ".cbor" if format == "cbor" else ".json")

        full_path = os.path.join(self.path, filename)

        if os.path.exists(full_path) and not force:
            raise FileExistsError(f"File {full_path} already exists. Use force=True to overwrite.")

        # Detect format
        if format == "auto":
            detected_format = self._detect_format(filename)
        else:
            detected_format = format
            if detected_format not in ["json", "cbor"]:
                raise FileFormatError(f"Invalid format: {format}. Use 'json', 'cbor', or 'auto'.")

        data_to_write = {"metadata": self.metadata, "records": self._records, "stats": {"record_count": self._record_count, "time_range": self._time_range, "schema_keys": list(self._schema_keys) if self._schema_keys else None}}

        self._write_data(data_to_write, full_path, detected_format)
        info(f"Successfully wrote {self._record_count} records to {full_path} ({detected_format} format)")
        return filename

    def read_from_file(self, filename: str) -> None:
        """
        Read metadata and log records from a file.

        Args:
            filename (str): The name of the file to read from.
        """
        full_path = os.path.join(self.path, filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        file_format = self._detect_format(filename)
        data_loaded = self._read_data(full_path, file_format)

        # Validate loaded data structure
        if not isinstance(data_loaded, dict):
            raise FileFormatError("Invalid file format: root must be a dictionary")

        if "metadata" not in data_loaded or "records" not in data_loaded:
            raise FileFormatError("Invalid file format: missing 'metadata' or 'records'")

        self.metadata = data_loaded.get("metadata", {})
        records = data_loaded.get("records", [])

        # Load stats if available (for performance optimization)
        stats = data_loaded.get("stats", {})
        self._record_count = stats.get("record_count", len(records))
        self._time_range = tuple(stats["time_range"]) if stats.get("time_range") else None
        schema_keys = stats.get("schema_keys")
        self._schema_keys = set(schema_keys) if schema_keys else None

        self.set_records(records)  # This will validate records if validation is enabled
        info(f"Successfully loaded {len(records)} records from {full_path} ({file_format} format)")

    def extract_by_time(self, time_min: int, time_max: int) -> List[Dict[str, Any]]:
        """
        Extract records with time values within a given range (inclusive).
        Optimized for performance.

        Args:
            time_min (int): The minimum time value.
            time_max (int): The maximum time value.

        Returns:
            List[Dict[str, Any]]: A list of records with 'time' between time_min and time_max.
        """
        # Early exit if no overlap with known time range
        if self._time_range:
            min_time, max_time = self._time_range
            if time_max < min_time or time_min > max_time:
                return []

        return [record for record in self._records if time_min <= record["time"] <= time_max]

    def extract_by_keys(self, keys: List[str]) -> List[Dict[str, Any]]:
        """
        Extract specific columns from each record.

        Args:
            keys (List[str]): The list of keys (columns) to extract.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing only the specified keys.
        """
        extracted = []
        for record in self._records:
            new_record = {}
            # Always include time unless explicitly omitted
            if "time" not in keys:
                new_record["time"] = record["time"]

            for key in keys:
                if key in record:
                    new_record[key] = record[key]
            extracted.append(new_record)
        return extracted

    def finalize(self, time_value: int, **summary_data: Any) -> None:
        """
        Append a final summary record containing overall statistics.

        Args:
            time_value (int): The timestamp for the summary record.
            **summary_data: Arbitrary keyword arguments representing summary data.
        """
        if not isinstance(time_value, int):
            raise ValidationError(f"Time value must be an integer, got {type(time_value)}")

        summary = {"summary": True, "time": time_value}
        summary.update(summary_data)

        # Don't validate summary records as they may have different schema
        old_validate = self.validate_schema
        self.validate_schema = False
        try:
            self._records.append(summary)
            self._record_count += 1
        finally:
            self.validate_schema = old_validate

    def extract_metadata(self) -> Dict[str, Any]:
        """Extract and return the global metadata for this logger."""
        return self.get_metadata()

    def extract_summary(self) -> List[Dict[str, Any]]:
        """Extract and return all final summary records from the log."""
        return [record for record in self._records if record.get("summary", False)]

    # New utility methods
    def get_record_count(self) -> int:
        """Get the total number of records."""
        return self._record_count

    def get_time_range(self) -> Optional[tuple]:
        """Get the time range (min, max) of all records."""
        return self._time_range

    def get_unique_keys(self) -> set:
        """Get all unique keys across all records."""
        if self._schema_keys and self.validate_schema:
            return self._schema_keys.copy()

        # Fallback: scan all records
        all_keys = set()
        for record in self._records:
            all_keys.update(record.keys())
        return all_keys

    def clear_records(self) -> None:
        """Clear all records and reset statistics."""
        self._records.clear()
        self._record_count = 0
        self._time_range = None
        self._schema_keys = None


# Example usage and testing
if __name__ == "__main__":
    # Create an instance of Logger with enhanced features
    logger = Logger("MyLogger", metadata={"version": "2.0", "description": "Enhanced log file"}, filename_pattern="{name}_v{metadata_hash}_{timestamp}", validate_schema=True)

    # Log some data with integer timestamps
    logger.log_data({"temperature": 23.5, "humidity": 60}, time_value=1000)
    logger.log_data({"temperature": 24.0, "humidity": 58}, time_value=1001)

    # Test various features
    print("Global Metadata:")
    print(logger.get_metadata())

    print(f"\nRecord count: {logger.get_record_count()}")
    print(f"Time range: {logger.get_time_range()}")
    print(f"Unique keys: {logger.get_unique_keys()}")

    print("\nAll Records:")
    for record in logger.get_records():
        print(record)

    # Test JSON format
    json_filename = logger.write_to_file(format="json")
    print(f"\nSaved to JSON: {json_filename}")

    # Test CBOR format if available
    if CBOR_AVAILABLE:
        cbor_filename = logger.write_to_file("test_data.cbor", format="cbor", force=True)
        print(f"Saved to CBOR: {cbor_filename}")

        # Test reading CBOR
        new_logger = Logger("TestRead")
        new_logger.read_from_file("test_data.cbor")
        print(f"Read from CBOR: {new_logger.get_record_count()} records")
    else:
        print("CBOR format not available (install cbor2)")

    # Add summary and test finalization
    logger.finalize(time_value=1002, avg_temp=23.75, total_readings=2)
    print(f"\nAfter finalization: {logger.get_record_count()} total records")
    print("Summary records:", logger.extract_summary())
