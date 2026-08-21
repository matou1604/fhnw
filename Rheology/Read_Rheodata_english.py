from pathlib import Path
import csv
import pandas as pd


# RheoCompass export labels. The parser accepts both English and German exports,
# but returns standardised English metadata column names.
METADATA_LABELS = {
    "Project:": "Project",
    "Projekt:": "Project",
    "Test:": "Test",
    "Versuch:": "Test",
}

RESULT_LABELS = {"Result:", "Ergebnis:"}
INTERVAL_LABELS = {"Interval and data points:", "Abschnitt und Datenpunkte:"}
INTERVAL_DATA_LABELS = {"Interval data:", "Abschnittsdaten:"}


def clean_cell(x):
    """Return a stripped, quote-free cell value."""
    if x is None:
        return ""
    return str(x).strip().strip('"').strip()


def first_nonempty_index(row):
    """Return the index of the first non-empty cell, or None."""
    for i, cell in enumerate(row):
        if clean_cell(cell) != "":
            return i
    return None


def first_nonempty(row):
    """Return the first non-empty cell, or an empty string."""
    idx = first_nonempty_index(row)
    return "" if idx is None else clean_cell(row[idx])


def read_text_with_fallback(path):
    """Read a CSV export while handling common RheoCompass encodings."""
    path = Path(path)
    raw = path.read_bytes()

    # UTF-16 BOM detection
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")

    # UTF-8 BOM detection
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")

    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown", b"", 0, 1, "Could not decode file with common encodings."
    )


def detect_delimiter(text):
    """Detect tab, semicolon, or comma delimiter from the first export rows."""
    sample = "\n".join(text.splitlines()[:20])

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t;,")
        return dialect.delimiter
    except csv.Error:
        # RheoCompass exports are commonly tab-separated.
        return "\t"


def align_row(cells, n_columns):
    """
    Align unit/data rows that can have leading empty cells.

    Example:
    ['', '1', '62.00', '232.09', ...]
    becomes
    ['1', '62.00', '232.09', ...]
    """
    cells = [clean_cell(cell) for cell in cells]

    while len(cells) > n_columns and cells[0] == "":
        cells = cells[1:]

    if len(cells) < n_columns:
        cells.extend([""] * (n_columns - len(cells)))

    return cells[:n_columns]


def combine_header_and_units(headers, units):
    """Create descriptive column names such as 'Storage Modulus [Pa]'."""
    columns = []

    for header, unit in zip(headers, units):
        header = clean_cell(header)
        unit = clean_cell(unit)
        columns.append(f"{header} {unit}" if unit else header)

    return columns


def convert_numeric_columns(df, min_numeric_fraction=0.6):
    """
    Convert mostly numeric columns to numeric dtype.

    Decimal dots and decimal commas are both supported. Text columns such as
    'Status' remain strings.
    """
    for column in df.columns:
        cleaned = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace("'", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        converted = pd.to_numeric(cleaned, errors="coerce")

        if converted.notna().mean() >= min_numeric_fraction:
            df[column] = converted

    return df


def _metadata_value(row, start_index):
    """Join all non-empty cells after a metadata label."""
    return " ".join(
        clean_cell(cell) for cell in row[start_index + 1:] if clean_cell(cell)
    )


def _number_from_cell(cell, field_name):
    """Convert a point/interval count to int with a useful error message."""
    try:
        return int(float(clean_cell(cell).replace(",", ".")))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not read {field_name} from value {cell!r}."
        ) from exc


def parse_rheology_csv(path):
    """
    Parse RheoCompass-style CSV exports in English or German.

    Accepted labels include:
        English: Project, Test, Result, Interval and data points, Interval data
        German:  Projekt, Versuch, Ergebnis, Abschnitt und Datenpunkte,
                 Abschnittsdaten

    Returns
    -------
    file_metadata : dict
        Standardised file-level metadata: Project and Test.

    sections : dict[tuple, pandas.DataFrame]
        One DataFrame per (Result_index, Interval).

    combined : pandas.DataFrame
        All datapoints combined, with standardised metadata columns:
        source_file, Project, Test, Result_index, Result, Interval,
        expected_points.
    """
    path = Path(path)
    text = read_text_with_fallback(path)
    delimiter = detect_delimiter(text)

    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    rows = [[clean_cell(cell) for cell in row] for row in rows]

    file_metadata = {}
    sections = {}

    current_result = ""
    current_result_index = 0
    i = 0

    while i < len(rows):
        row = rows[i]
        first_index = first_nonempty_index(row)

        if first_index is None:
            i += 1
            continue

        first = clean_cell(row[first_index])

        # File-level metadata: Project / Test (and German equivalents)
        if first in METADATA_LABELS:
            file_metadata[METADATA_LABELS[first]] = _metadata_value(row, first_index)
            i += 1
            continue

        # Result-level metadata: Result / Ergebnis
        if first in RESULT_LABELS:
            current_result_index += 1
            current_result = _metadata_value(row, first_index)
            i += 1
            continue

        # Interval block: Interval and data points / Abschnitt und Datenpunkte
        if first in INTERVAL_LABELS:
            if len(row) < first_index + 3:
                raise ValueError(
                    f"Malformed interval row in {path.name}: expected interval "
                    f"number and data-point count after {first!r}."
                )

            interval_number = _number_from_cell(
                row[first_index + 1], "interval number"
            )
            expected_points = _number_from_cell(
                row[first_index + 2], "expected data-point count"
            )

            # Locate the paired header row: Interval data / Abschnittsdaten.
            i += 1
            while i < len(rows) and first_nonempty(rows[i]) not in INTERVAL_DATA_LABELS:
                i += 1

            if i >= len(rows):
                raise ValueError(
                    f"Could not find 'Interval data:' after interval "
                    f"{interval_number} in {path.name}."
                )

            header_row = rows[i]
            header_start = first_nonempty_index(header_row)
            headers = [clean_cell(cell) for cell in header_row[header_start + 1:]]

            # Remove blank columns exported at the right edge.
            while headers and headers[-1] == "":
                headers.pop()

            if not headers:
                raise ValueError(
                    f"No measurement headers found for interval {interval_number} "
                    f"in {path.name}."
                )

            n_columns = len(headers)

            # The next non-empty row is the units row.
            i += 1
            while i < len(rows) and first_nonempty_index(rows[i]) is None:
                i += 1

            if i >= len(rows):
                raise ValueError(
                    f"No units row found for interval {interval_number} in {path.name}."
                )

            units = align_row(rows[i], n_columns)
            columns = combine_header_and_units(headers, units)

            # Read data until the next Result or Interval block.
            data = []
            i += 1
            while i < len(rows):
                current_row = rows[i]
                row_first = first_nonempty(current_row)

                if row_first in RESULT_LABELS or row_first in INTERVAL_LABELS:
                    break

                if first_nonempty_index(current_row) is None:
                    i += 1
                    continue

                aligned = align_row(current_row, n_columns)

                # Valid measurement rows start with the numeric Point No.
                try:
                    float(aligned[0].replace(",", "."))
                except ValueError:
                    i += 1
                    continue

                data.append(aligned)
                i += 1

            df = pd.DataFrame(data, columns=columns)
            df = convert_numeric_columns(df)

            # Add standardised metadata as real DataFrame columns.
            df.insert(0, "source_file", path.name)
            df.insert(1, "Project", file_metadata.get("Project", ""))
            df.insert(2, "Test", file_metadata.get("Test", ""))
            df.insert(3, "Result_index", current_result_index)
            df.insert(4, "Result", current_result)
            df.insert(5, "Interval", interval_number)
            df.insert(6, "expected_points", expected_points)

            sections[(current_result_index, interval_number)] = df
            continue

        i += 1

    combined = (
        pd.concat(sections.values(), ignore_index=True)
        if sections
        else pd.DataFrame()
    )

    return file_metadata, sections, combined


def read_rheodata_folder(folder, pattern="*.csv"):
    """Read all matching RheoCompass exports from a folder."""
    folder = Path(folder)

    all_data = []
    all_metadata = {}
    all_sections = {}

    for path in folder.glob(pattern):
        file_metadata, sections, df_all = parse_rheology_csv(path)

        all_metadata[path.name] = file_metadata
        all_sections[path.name] = sections

        if not df_all.empty:
            all_data.append(df_all)

    combined = (
        pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    )

    return all_metadata, all_sections, combined


def read_data(path):
    """Backward-compatible single-file reader used by existing notebooks."""
    return parse_rheology_csv(path)
