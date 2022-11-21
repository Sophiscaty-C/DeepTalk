from graia.ariadne.app import Ariadne
from graia.ariadne.connection.config import config
from graia.ariadne.model import Friend

app = Ariadne(config(verify_key="1234567890", account=2890921034))


@app.broadcast.receiver("FriendMessage")
async def friend_message_listener(app: Ariadne, friend: Friend):
    await app.send_message(friend, "Hello, World!")


Ariadne.launch_blocking()