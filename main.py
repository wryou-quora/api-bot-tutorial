# Welcome to the Poe API tutorial. The starter code provided provides you with a quick way to get
# a bot running. By default, the starter code uses the EchoBot, which is a simple bot that echos
# a message back at its user and is a good starting point for your bot, but you can
# comment/uncomment any of the following code to try out other example bots.

from fastapi_poe import make_app

from all_caps import AllCapsBot
from battle import BattleBot
from catbot import CatBot
from concurrent_battle import ConcurrentBattleBot
from echobot import EchoBot
from langcatbot import LangCatBot
from three_talkers import ThreeTalkers


OPENAI_API_KEY = "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"

# Echo bot is a very simple bot that just echoes back the user's last message.
# bot = EchoBot()

# A sample bot that showcases the capabilities the protocol provides. Please see the
# following link for the full set of available message commands:
# https://github.com/poe-platform/api-bot-tutorial/blob/main/catbot/catbot.md
# bot = CatBot()

# A custom chatbot built on top of ChatGPT and LangChain.
# Add your OpenAI key here, e.g. sk-1234
# You can obtain a key at https://platform.openai.com/account/api-keys
# bot = LangCatBot(OPEN_AI_API_KEY)

# A bot that wraps Poe's Sage bot, but makes all messages ALL CAPS.
# bot = AllCapsBot()

# A bot that calls on both Sage and Claude-Instant and shows the results.
# bot = BattleBot()

# Like BattleBot, but streams both responses at once.
# bot = ConcurrentBattleBot()

# Optionally add your Poe API key here. You can go to https://poe.com/create_bot?api=1 to generate
# one. We strongly recommend adding this key for a production bot to prevent abuse,
# but the starter example disables the key check for convenience.
# POE_API_KEY = ""
# app = make_app(bot, api_key=POE_API_KEY)

bot = ThreeTalkers(openai_api_key=OPENAI_API_KEY)
app = make_app(bot, allow_without_key=True)
