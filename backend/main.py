from backend.app.api import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = app.state.config.server
    uvicorn.run(app, host=config.host, port=config.port)
