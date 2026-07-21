import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Small websocket client for VA inference servers."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(
        self,
    ) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, OSError) as exc:
                logging.info("Still waiting for server... (%s)", exc)
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(
        self,
        prompt: Optional[str] = None,
        episode_tag: Optional[str] = None,
        episode_name: Optional[str] = None,
    ) -> Dict:
        payload: Dict = {"reset": True}
        if prompt is not None:
            payload["prompt"] = prompt
        if episode_tag is not None:
            payload["episode_tag"] = episode_tag
        if episode_name is not None:
            payload["episode_name"] = episode_name
        return self.infer(payload)

    def flush_pred_video(self) -> Dict:
        return self.infer({"flush_pred_video": True})
