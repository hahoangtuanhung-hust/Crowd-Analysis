import os

from backend.app.api import create_app
from backend.app.core.config import load_config

CONFIG_PATH = os.environ.get("CROWD_CONFIG", "configs/default.yaml")
app = create_app(load_config(CONFIG_PATH))


if __name__ == "__main__":
    import uvicorn

    config = app.state.config.server
    uvicorn.run(app, host=config.host, port=config.port)
