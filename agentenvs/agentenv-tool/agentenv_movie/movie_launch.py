"""
Entrypoint for the movie agent environment.
"""

import argparse
import multiprocessing
import uvicorn

from .movie_utils import debug_flg

def run_server(port, host):
    """Function to run a single uvicorn server instance."""
    uvicorn.run("agentenv_movie:app", host=host, port=port, reload=debug_flg, workers=1)

def launch():
    """Entrypoint for `alfworld` command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Starting port for the server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address to bind to.")
    parser.add_argument("--instances", type=int, default=1, help="Number of server instances to run.")
    args = parser.parse_args() 

    # List to hold processes
    processes = []

    # Launch multiple server instances
    for i in range(args.instances):
        port = args.port + i  # Increment port for each instance
        process = multiprocessing.Process(target=run_server, args=(port, args.host))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

if __name__ == "__main__":
    launch()