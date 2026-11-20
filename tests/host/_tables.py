##########################################################################################
# tests/host/_tables.py: small PDS3 index tables for the Host tests
##########################################################################################

from pathlib import Path


def write_table(directory: Path, stem: str, columns: dict[str, list[str]], *,
                label_rows: int | None = None,
                integers: tuple[str, ...] = ()) -> Path:
    """Write an ASCII table and its detached PDS3 label; return the label's path.

    Parameters:
        directory: The directory to write into.
        stem: The file name without its extension.
        columns: The values of each column, by column name, as text.
        label_rows: The number of rows that the label states; by default, the number of
            rows written.
        integers: The columns that the label describes as ASCII_INTEGER; every other
            column is CHARACTER.
    """

    names = list(columns)
    widths = [max(len(value) for value in columns[name]) for name in names]
    nrows = len(columns[names[0]])
    row_bytes = sum(widths) + len(widths) - 1 + 2
    lines = [','.join(columns[name][i].ljust(width)
                      for (name, width) in zip(names, widths, strict=True))
             for i in range(nrows)]
    (directory / f'{stem}.tab').write_bytes(''.join(line + '\r\n'
                                                    for line in lines).encode())

    rows = nrows if label_rows is None else label_rows
    start = 1
    objects = []
    for (name, width) in zip(names, widths, strict=True):
        data_type = 'ASCII_INTEGER' if name in integers else 'CHARACTER'
        objects += ['  OBJECT = COLUMN', f'    NAME = {name}',
                    f'    DATA_TYPE = {data_type}', f'    START_BYTE = {start}',
                    f'    BYTES = {width}', '  END_OBJECT = COLUMN']
        start += width + 1
    label = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
             f'RECORD_BYTES = {row_bytes}', f'FILE_RECORDS = {rows}',
             f'^INDEX_TABLE = "{stem}.tab"', 'OBJECT = INDEX_TABLE',
             '  INTERCHANGE_FORMAT = ASCII', f'  ROWS = {rows}',
             f'  ROW_BYTES = {row_bytes}', f'  COLUMNS = {len(names)}',
             *objects, 'END_OBJECT = INDEX_TABLE', 'END', '']
    filepath = directory / f'{stem}.lbl'
    filepath.write_text('\n'.join(label))
    return filepath

##########################################################################################
