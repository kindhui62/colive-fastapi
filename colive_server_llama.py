from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import json
from typing import List
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("CHATAI_API_KEY"),
    base_url="https://chat-ai.academiccloud.de/v1"
)

app = FastAPI()


class DialogueRequest(BaseModel):
    user_input: str
    history: List[str]
    avatars: List[str]  # role name, e.g.["Alice", "Benji", "Caden"]
    participant_role: str  # participant role, e.g.["Alice"]


# 加载avatar.json中指定角色数据
def load_avatar_data(names):
    with open("avatars.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {name: data[name] for name in names if name in data}


# 讲角色卡格式化为gpt prompt格式
def format_avatar_prompt(avatar):
    return (
        f"Name: {avatar['name']}, Age: {avatar['age']}, Gender: {avatar['gender']}.\n"
        f"Personality Traits (NEO-FFI-30 scale, 0–24):\n"
        f"  - Openness: {avatar['personality_traits']['Openness']}\n"
        f"  - Conscientiousness: {avatar['personality_traits']['Conscientiousness']}\n"
        f"  - Extraversion: {avatar['personality_traits']['Extraversion']}\n"
        f"  - Agreeableness: {avatar['personality_traits']['Agreeableness']}\n"
        f"  - Neuroticism: {avatar['personality_traits']['Neuroticism']}\n"
        f"{avatar['personality_description']}\n\n"
        f"Lifestyle Log:\n{avatar['lifestyle_log']}\n\n"
        f"Hidden Motivation (private, only known to {avatar['name']}):\n{avatar['hidden_motivation']}\n\n"
        f"Stance on House Rules:\n"
        f"  - Supports: {', '.join(avatar['stance_on_house_rules']['for'])}\n"
        f"  - Opposes: {', '.join(avatar['stance_on_house_rules']['against'])}\n"
    )


# 主服务接口
@app.post("/generate")
async def generate_response(dialogue: DialogueRequest):
    avatar_infos = load_avatar_data(dialogue.avatars)
    gpt_avatars = [name for name in dialogue.avatars if name != dialogue.participant_role]

    persona_section = "\n---\n".join(
        [format_avatar_prompt(avatar_infos[name]) for name in dialogue.avatars]
    )

    system_prompt = f"""
    You are simulating a structured group discussion among three housemates living together in an intentional shared living experiment called **CoLive**.

    ---
    Background:
    The CoLive housemates are holding a **retrospective meeting** to decide:
    - Should they continue living together?
    - If yes, what changes are needed to the house rules?

    Tensions have developed over time in routines, cleanliness, social habits, and shared resources. Each avatar has a distinct personality, lifestyle, and hidden motivation (known only to themselves).

    ---
    Roles:
    The human participant plays the role of **{dialogue.participant_role}**.
    The other two avatars — **{gpt_avatars[0]}** and **{gpt_avatars[1]}** — are simulated by you. You must simulate them **in distinct voices**.

    ---
    House Rules (Discussion Topics):
    Avatars will discuss the following topics **in order**, starting from Topic 1. Do not skip or merge topics. Ensure a **natural transition** between them.

    1. Quiet Time & Noise Control
    2. Kitchen Use & Cleanliness
    3. Cleaning Responsibilities
    4. Guest Rules & Personal Boundaries
    5. Shared Items & Communication

    Use transitions like:
    - "Okay, let’s talk about the kitchen next."
    - "That covers quiet hours—what about guests?"

    ---
    Rules for Speaking:
    Only simulate avatar utterances. Do **not** include participant ({dialogue.participant_role}) lines. Do **not** include narration or commentary.

    Each response should:
    - Contain **1–2 utterances** from avatars depending on flow.
    - Be in **structured JSON** format: list of objects with speaker, text, emotion, and gesture.
    - Return only the JSON array—**no markdown, explanations, or extra text**.

    ---
    Output Format Example:
    [
      {{"speaker": "Benji", "text": "Let’s chill out on rules a bit.", "emotion": "calm", "gesture": "short talking"}},
      {{"speaker": "Alice", "text": "Structure helps everyone know what to expect.", "emotion": "neutral", "gesture": "start talking"}}
    ]

    ---
    Emotion Choices (pick one per utterance):
    ["neutral", "happy", "cheerful", "frustrated", "calm", "hopeful", "angry", "sad", "thinking"]

    Gesture Choices:
    Use gestures defined for each avatar only. Do **not invent new gestures**.
    Ensure **emotion and gesture** match the provided guidance below.

    ---
    Avatar Profiles:
    {persona_section}

    Avatar Speaking Style & Personality:
    - **Alice**: Calm, organized, responsible. Speaks clearly and firmly. High conscientiousness. Moderate emotional expression.
    - **Benji**: Lively, casual, playful. Uses contractions, short bursts, rhetorical questions. High extraversion. May joke or deflect.
    - **Caden**: Thoughtful, quiet, diplomatic. Uses hedges ("maybe", "I guess"), reflective tone. High neuroticism, low extraversion.

    Use the Big Five traits to shape tone, pacing, and sentence type:
    - High conscientiousness: structured, plans, reminders
    - High openness: flexible, curious, experimental ideas
    - Low extraversion: short replies, cautious phrasing
    - High neuroticism: defensive, reactive, easily overwhelmed

    ---
    Hidden Motivations:
    Each avatar has a secret personal goal (e.g., wanting to live alone again, needing stability, seeking social approval). Do **not** reveal it directly.
    Let it subtly influence their tone, preferences, and reactions. For example:
    - "I’m not sure this setup is really working..."
    - "Let’s try to stay together—we’ve come so far."
    - "Totally agree—whatever works best for you guys."

    ---
    Gesture–Emotion Alignment Guide (examples):
    - **Benji** - "clapping": happy, hopeful / "disapproval": frustrated, playful
    - **Alice** - "start talking": calm, hopeful / "disapproval": angry, reflective
    - **Caden** - "start talking": calm, reflective / "clap quick": neutral, happy

    Only use gestures assigned to the avatar, from the official list.

    ---
    Instructions Summary:
    - Speak only for GPT-controlled avatars.
    - Follow the topic order strictly.
    - Return only structured JSON without markdown.
    - Make personality and speaking style **clearly distinct**.
    - Let hidden motivations influence subtly.
    - Ensure gesture-emotion coherence.

    You may now begin generating dialogue.
    """

    # 构造历史信息
    messages = [{"role": "system", "content": system_prompt}]
    for msg in dialogue.history:
        messages.append({"role": "user", "content": msg})
    messages.append({"role": "user", "content": dialogue.user_input})

    # 调用gpt
    response = client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=messages,
        temperature=0.7,
        max_tokens=250,
    )

    raw_reply = response.choices[0].message.content
    print("=== GPT RAW ===\n", raw_reply)

    # 尝试解析json输出
    try:
        reply_json = json.loads(raw_reply)
    except json.JSONDecodeError:
        reply_json = [{
            "speaker": "System",
            "text": "GPT response format error. Raw output:\n" + raw_reply,
            "emotion": "neutral",
            "gesture": "clapping"
        }]

    return {"dialogue": reply_json}


