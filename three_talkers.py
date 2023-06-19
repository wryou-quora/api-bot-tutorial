"""

Three Talkers. Moderates the chat and suggests other talkers in the chat.

"""
from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import AsyncIterable, AsyncIterator, Sequence

from fastapi_poe import PoeBot, run
from fastapi_poe.client import BotMessage, MetaMessage, stream_request
from fastapi_poe.types import ProtocolMessage, QueryRequest
from sse_starlette.sse import ServerSentEvent

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


moderator_prompt = """You are a chat moderator. \
Answer always starts only with YES or NO. If it's ambiguous, answer NO. Put your reasons after that. \
You decide if given set of messages could be either of followings: \
* harmful \
* offensive \
* hateful \
* discriminative \
if they were sent to the human recipent. 
"""

conversational_prompt = """You are a friendly bot who wants to continue the conversation with the user. \
Discover the user's interest in the conversation.
"""


class ThreeTalkers(PoeBot):
    openai_api_key: str = ""
    user_block_table: dict[str, bool] = defaultdict(lambda: False)

    def __init__(self, openai_api_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openai_api_key = openai_api_key

    def is_harmful(self, query: QueryRequest) -> bool:
        messages = [SystemMessage(content=moderator_prompt)]
        for i, message in enumerate(query.query[-6:]):
            if message.role == "user":
                print(i, message.role, message.content)
                messages.append(HumanMessage(content=message.content))
        chat = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0,
        )
        res = chat(messages)
        print(res)
        return res.content.startswith("YES")

    async def get_response(self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        self.user_block_table[query.user_id] |= self.is_harmful(query)
        print(f"is {query.user_id} blocked: {self.user_block_table[query.user_id]}")
        if self.user_block_table[query.user_id]:
            yield self.text_event("Your chat is suspended due to the harmful message.")
        else:
            messages = [SystemMessage(content=conversational_prompt)]
            for message in query.query:
                if message.role == "bot":
                    messages.append(AIMessage(content=message.content))
                elif message.role == "user":
                    messages.append(HumanMessage(content=message.content))
            handler = AsyncIteratorCallbackHandler()
            chat = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                streaming=True,
                callback_manager=AsyncCallbackManager([handler]),
                temperature=0,
            )
            asyncio.create_task(chat.agenerate([messages]))
            async for token in handler.aiter():
                yield self.text_event(token)

if __name__ == "__main__":
    run(ThreeTalkers())