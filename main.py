import os
from creart import create
from graia.saya import Saya
from graia.broadcast import Broadcast
from graia.ariadne.app import Ariadne
from graia.ariadne.connection.config import config, WebsocketClientConfig

create(Broadcast)
saya = create(Saya)
app = Ariadne(config(2890921034, "1234567890", WebsocketClientConfig("http://localhost:8080")))
ignore = ["__init__.py", "__pycache__"]

with saya.module_context():
    for module in os.listdir("modules"):
        if module in ignore:
            continue
        try:
            if os.path.isdir(module):
                saya.require(f"modules.{module}")
            else:
                saya.require(f"modules.{module.split('.')[0]}")
        except ModuleNotFoundError:
            pass

app.launch_blocking()