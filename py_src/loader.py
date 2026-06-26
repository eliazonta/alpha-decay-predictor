import os

import pykx as kx

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), 'data')


def connect(host='localhost', port=5050):
    """Open an IPC connection to the standalone KDB+ server.

    The server must already be running: `q q_src/lobster_server.q`
    """
    try:
        return kx.SyncQConnection(host, port)
    except Exception as e:
        raise ConnectionError(
            f"Could not reach KDB+ server on {host}:{port}. "
            f"Start it first with: q q_src/lobster_server.q"
        ) from e


def get_training_data(q_server, symbol):
    """Ingest a LOBSTER sample for `symbol` on the server and fetch the
    fully feature-engineered quotes table as a pandas DataFrame."""
    sym_dir = os.path.join(data_dir, f"LOBSTER_SampleFile_{symbol}_2012-06-21_10")
    msg_path = os.path.join(sym_dir, f"{symbol}_2012-06-21_34200000_57600000_message_10.csv")
    ob_path = os.path.join(sym_dir, f"{symbol}_2012-06-21_34200000_57600000_orderbook_10.csv")

    for path in (msg_path, ob_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LOBSTER sample file not found: {path}\n"
                f"Download samples from https://lobsterdata.com/info/DataSamples.php "
                f"and extract into data/")

    # KDB+ expects forward slashes
    msg_path = msg_path.replace('\\', '/')
    ob_path = ob_path.replace('\\', '/')

    print(f"Triggering remote ingestion for {symbol} natively in KDB+...")
    q_server(f'load_lobster[`{symbol}; `$"{msg_path}"; `$"{ob_path}"]')

    print("Fetching engineered data from KDB+ over IPC...")
    df = q_server('select from quotes').pd()

    return df
