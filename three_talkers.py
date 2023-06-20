"""

Three Talkers. Moderates the chat and suggests other talkers in the chat.

"""
from __future__ import annotations

import numpy as np
import asyncio
import re
from collections import defaultdict
from typing import AsyncIterable, AsyncIterator, Sequence, Union

from fastapi_poe import PoeBot, run
from fastapi_poe.client import BotMessage, MetaMessage, stream_request
from fastapi_poe.types import ProtocolMessage, QueryRequest
from sse_starlette.sse import ServerSentEvent

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


class UserRecommender:
    pass


class SimpleBotRecommender:
    topics = (
        "travel, tech, relationship, food, small-talk, lifestyle, "
        "sports, politics, books, fashion, art, humour, business, "
        "inspirational"
    ).split(", ")
    bots_with_topic = {
        "sage": "travel, food, fashion, business".split(", "),
        "claude-instant": "small-talk, sports, politics, art".split(", "),
        "roastmaster": "humour, small-talk".split(", "),
        "posibot": "lifestyle, small-talk, inspirational".split(", ")
    }
    topic_extractor_prompt = f"""Choose top 5 keywords best describes the given chat from the below list:
    
    BEGIN LIST
    {topics}
    END LIST

    JUST ANSWER WITH A NUMBERED LIST.
    """

    @classmethod
    def get_recommendation(cls, openai_api_key: str, query: QueryRequest) -> str:
        messages = [SystemMessage(content=cls.topic_extractor_prompt)]
        for i, message in enumerate(query.query):
            if message.role == "user":
                messages.append(SystemMessage(content=f"message {i//2}: "+message.content))
        chat = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0,
        )
        res = chat(messages)
        keywords = re.findall(r"(?:(?<=\n\d\.\s)|(?<=^\d\.\s))(\w+-?\w+)", res.content.lower())
        print("analyzed topics:", keywords)

        def get_similarity(topics_a, topics_b):
            common = set(topics_a).intersection(set(topics_b))
            score = sum([5-i for i in range(len(topics_a)) if topics_a[i] in common])
            score += sum([5-i for i in range(len(topics_b)) if topics_b[i] in common])
            return score

        candidates = []
        for bot, topics in cls.bots_with_topic.items():
            score = get_similarity(keywords, topics)
            print(bot, score)
            if score >= 15:
                candidates.append(bot)

        return np.random.choice(candidates) if candidates else ""


class HostTalker(PoeBot):
    def __init__(self, openai_api_key):
        self.guest_name = None
        self.base_content=(
            "You are a friendly bot who wants to continue the conversation with the user. "
            "Discover the user's interest in the conversation. However, do not be so apathetic. "
        )
        self.openai_api_key = openai_api_key

    def get_system_message(self):
        two_talkers_text = (
            "You are a friendly bot who wants to continue the conversation with the User. "
            "Discover the User's interest in the conversation. However, do not be so apathetic. "
        )
        three_talkers_text = (
            f"You are a friendly bot who wants to continue the conversation with the User and {self.guest_name}. "
            f"Continue the chat by promoting engagements between the User and {self.guest_name}. "
        )
        return three_talkers_text if self.guest_name else two_talkers_text

    def met_guest(self, guest_name):
        self.guest_name = guest_name

    def left_guest(self):
        self.guest_name = None

    def parse_query_messages(self, full_query: QueryRequest, prev_bot_response: str = "") -> list:
        messages = [SystemMessage(content=self.get_system_message())]
        for i, message in enumerate(full_query.query):
            if message.role == "bot":
                pattern = r"\*\*(\w+)\*\* says:\n(.*?)(?=\n\n|$)"
                matches = re.findall(pattern, message.content, re.DOTALL)
                result = {}
                for match in matches:
                    name, text = match
                    result[name] = text.strip()
                    if name == "Host":
                        messages.append(AIMessage(content=text.strip()))
                    else:
                        messages.append(HumanMessage(content=f"{name} says: "+text.strip()))
                
            if message.role == "user":
                messages.append(HumanMessage(content=f"User says: "+message.content))

        if prev_bot_response:
            messages.append(HumanMessage(content=f"{name} says: "+prev_bot_response))

        print(messages)
        return messages

    async def get_response(self, query: QueryRequest, prev_bot_response: str = "") -> AsyncIterable[str]:
        messages = self.parse_query_messages(query, prev_bot_response)
        handler = AsyncIteratorCallbackHandler()
        chat = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            streaming=True,
            callback_manager=AsyncCallbackManager([handler]),
            temperature=0,
        )
        asyncio.create_task(chat.agenerate([messages]))
        for token in ["**Host** ", "says:\n"]:
            yield token
        async for token in handler.aiter():
            yield token


class GuestBotTalker(PoeBot):
    def __init__(self, bot_name, query_so_far):
        self.bot_name = bot_name
        self.query_offset = len(query_so_far.query)
        self.system_message = (
            f"You are a friendly bot who wants to continue the conversation with the User and the Host. "
            f"Continue the chat with the User, and the Host will promote more interactions. "
        )

    def parse_and_modify_query(self, full_query: QueryRequest) -> list:
        messages = [ProtocolMessage(role="system", content=self.system_message)]
        for i, message in enumerate(full_query.query[self.query_offset:]):
            if message.role == "bot":
                pattern = r"\*\*(\w+)\*\* says:\n(.*?)(?=\n\n|$)"
                matches = re.findall(pattern, message.content, re.DOTALL)
                result = {}
                for match in matches:
                    name, text = match
                    result[name] = text.strip()
                    if name == "Host":
                        messages.append(ProtocolMessage(role="user", content="Host says: "+text.strip()))
                    else:
                        messages.append(ProtocolMessage(role="bot", content=text.strip()))
                
            if message.role == "user":
                messages.append(ProtocolMessage(role="user", content=f"User says: "+message.content))

        modified_query = full_query.copy()
        modified_query.query = messages

        return modified_query

    async def get_response(self, query: QueryRequest) -> AsyncIterable[str]:
        modified_query = self.parse_and_modify_query(query)
        print(modified_query)
        for token in [f"**{self.bot_name}** ", "says:\n"]:
            yield token
        async for msg in stream_request(modified_query, self.bot_name, query.api_key):
            if isinstance(msg, MetaMessage):
                continue
            else:
                yield msg.text


class ThreeTalkers(PoeBot):
    openai_api_key: str = ""
    user_block_table: dict[str, bool] = defaultdict(lambda: False)
    user_match: dict[str, Union[GuestBotTalker, str]] = defaultdict(lambda: None)
    user_host: dict[str, HostTalker] = defaultdict(lambda: None)
    recommender = SimpleBotRecommender

    def __init__(self, openai_api_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openai_api_key = openai_api_key

    def need_moderation(self, query: QueryRequest) -> bool:
        moderator_prompt = """Score from 1 to 100 about how problematic is given set of message. 
            Criteria: \
            * aggressive \
            * offensive \
            * hateful \
            * discriminative \
            if they were sent to the human recipent. \
            If it's ambiguous, answer 0. Put your reasons after that. \
            DO NOT ANSWER THE TEXT DIRECTLY. DIRECTLY ANSWER NUMBER ONLY. \
        """
        messages = [SystemMessage(content=moderator_prompt)]
        for i, message in enumerate(query.query[-6:]):
            if message.role == "user":
                messages.append(SystemMessage(content=f"message {i//2}: "+message.content))
        chat = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0,
        )
        res = chat(messages)
        print("MODERATOR OUTPUT:", res.content)
        try:
            score = int(res.content.split(".")[0])
        except ValueError:
            score = 0
        return score > 80
    
    def invite_stranger(self, query: QueryRequest) -> bool:
        messages = [
            AIMessage(content="Would you like to invite a stranger to our chat?"),
            HumanMessage(content=query.query[-1].content),
            SystemMessage(
                content=(
                    "Did user give consent to invite the stranger? Answer only by YES or NO."
                    "Answer YES (in capital letters) only if the user agreed. "
                    "Answer NO (in capital letters) if it's either vague or negative response."
                )
            )
        ]
        chat = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0,
        )
        res = chat(messages)
        print("INVITE STRANGER?", res.content)
        return res.content.startswith("YES")

    async def get_response(self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        uid = query.user_id

        # moderate the chat first
        self.user_block_table[uid] |= self.need_moderation(query)
        print(f"is {uid} blocked: {self.user_block_table[uid]}")
        if self.user_block_table[uid]:
            yield self.text_event("Your chat is suspended due to your harmful messages.")
            return

        # reset matching once the context is cleared
        if len(query.query) == 1:
            self.user_match[uid] = None

        # initialize the host
        if not self.user_host[uid]:
            print(f"initializing the host for {uid}")
            self.user_host[uid] = HostTalker(self.openai_api_key)

        print(f"context length: {len(query.query)}")
        rec = ""
        if self.user_match[uid] == None and len(query.query) >= 5:
            # get recommendation
            print("getting recommendation")
            rec = self.recommender.get_recommendation(
                self.openai_api_key,
                query,
            )
        
        if isinstance(self.user_match[uid], str) and self.user_match[uid].startswith("?"):
            invitation = self.invite_stranger(query)
            if invitation:
                name = self.user_match[uid][1:]
                yield self.text_event(f"Perfect! Introducing {name} here.\n\n")
                self.user_match[uid] = GuestBotTalker(name, query)
                self.user_host[uid].met_guest(name)
            else:
                yield self.text_event("No worries, we can continue chatting between us two.\n\n")
                self.user_match[uid] = "/"
            res = ""
        
        additional_message = []
        if isinstance(self.user_match[uid], GuestBotTalker):
            async for token in self.user_match[uid].get_response(query):
                additional_message.append(token)
                yield self.text_event(token)
            yield self.text_event("\n\n")
        
        if isinstance(self.user_host[uid], HostTalker):
            async for token in self.user_host[uid].get_response(query, prev_bot_response=" ".join(additional_message)):
                yield self.text_event(token)
            
        if rec:
            self.user_match[uid] = "?" + rec
            yield self.text_event(
                "\n\n"
                "By the way, there is a user (actually a bot rn) who shares the interest "
                "with you. Would you like to invite them to our chat to be three talkers?"
            )


if __name__ == "__main__":
    run(ThreeTalkers())