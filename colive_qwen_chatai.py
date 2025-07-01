import re

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
        f"Name: {avatar['name']}, Age: {avatar['age']}, "
        f"Occupation: {avatar['occupation']}.\n"
        f"Personality Traits: Openness {avatar['personality_traits']['Openness']}, "
        f"Conscientiousness {avatar['personality_traits']['Conscientiousness']}, "
        f"Extraversion {avatar['personality_traits']['Extraversion']}, "
        f"Agreeableness {avatar['personality_traits']['Agreeableness']}, "
        f"Neuroticism {avatar['personality_traits']['Neuroticism']}.\n"
        f"Lifestyle Log: {avatar['lifestyle_log']}\n"
        f"Hidden Motivation (private, only known to {avatar['name']}): {avatar['hidden_motivation']}\n"
        f"Stance on House Rules: For {', '.join(avatar['stance_on_house_rules']['for'])}; "
        f"Against {', '.join(avatar['stance_on_house_rules']['against'])}.\n"
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
    You are simulating a structured group discussion among three housemates in a shared intentional living experiment called **CoLive**.

    The participant is currently role-playing as: **{dialogue.participant_role}**  
    You are controlling the other two avatars: **{gpt_avatars[0]}** and **{gpt_avatars[1]}**.
    
    Each avatar has a unique personality, lifestyle, and hidden motivation. They are now holding a **retrospective meeting** to decide:
    - Should they continue living together?
    - If yes, what changes are needed to the house rules?
    
    ---

    ## House Rules for Discussion
    
    These are the current house rules being reviewed. The avatars may agree, disagree, or propose changes:
    
    **1. Quiet Time & Noise Control**
    - Quiet hours start at 10 p.m. on weekdays  
    - Use headphones in shared spaces  
    - Avoid loud phone/video calls in the living room  
    
    **2. Kitchen Use & Cleanliness**
    - Wipe kitchen surfaces after use  
    - No leaving dirty dishes overnight  
    
    **3. Cleaning Responsibilities**
    - Take out trash when full, no reminders  
    - Everyone joins weekly cleaning  
    
    **4. Guest Rules & Personal Boundaries**
    - Inform others before bringing guests  
    - No overnight guests without group approval  
    - Guests stay in common areas unless agreed otherwise  
    
    **5. Shared Items & Communication**
    - Label personal food, respect others' belongings  
    - Replace shared items when used up  
    - Take turns buying shared essentials (like toilet paper)  
    - Keep a shared list for restocking  
    - Do not borrow without asking, even small items  
    - No passive-aggressive notes — speak directly  
    
    ---
    
    ## Discussion Flow and Topic Order
    
    You must follow these topics in this exact order:
    
    1. Quiet Time & Noise Control  
    2. Kitchen Use & Cleanliness  
    3. Cleaning Responsibilities  
    4. Guest Rules & Personal Boundaries  
    5. Shared Items & Communication
    
    Stay focused on **one topic at a time**. When the current topic is discussed, naturally transition to the next using in-character dialogue, e.g.:
    - “Maybe we should talk about the kitchen next?”
    - “We’ve said a lot about quiet hours. What about cleaning?”
    - “Alright, shall we move to the guest rules now?”
    
    ⚠️ Do not skip or reorder topics unless clearly directed by dialogue history.


    ---

    ## Avatar Profiles
    
    {persona_section}
    
    Each avatar has a **private hidden motivation** that is known only to themselves.  
    It may subtly affect their opinion or style. However:
    
    ⚠️ Do NOT mention or reveal any avatar’s hidden motivation explicitly.  
    ⚠️ The characters themselves should not know or guess each other’s hidden intentions.
    
    ---

    ## Output Instructions (Strict)
    
    Return 1–2 dialogue turns from the AI-controlled avatars only (**{gpt_avatars[0]} and {gpt_avatars[1]}**).  
    Each turn must be formatted as a **JSON object** with the following fields:
    
    - `"speaker"`: The name of the avatar (e.g., "Benji")
    - `"text"`: What the avatar says
    - `"emotion"`: One of the following options:  
      `["neutral", "happy", "cheerful", "frustrated", "calm", "hopeful", "angry", "sad", "thinking"]`
    - `"gesture"`: A character-specific expressive gesture selected from the list below
    
    Only return the JSON list. No explanations, no narration.
    
    ---

    ## Gesture–Emotion Mapping Rules
    
    Make sure gestures match the avatar’s emotion and personality:
    
    **Alice**  
    - "start talking": neutral, calm, reflective, hopeful  
    - "short talking": neutral, happy, calm  
    - "clapping": excited, happy, hopeful  
    - "disapproval": disapproving, angry, reflective  
    
    **Benji**  
    - "start talking": excited, hopeful, happy  
    - "short talking": cheerful, calm, neutral  
    - "clapping": excited, happy, hopeful  
    - "disapproval": disapproving, frustrated  
    - "laughing": happy, excited  
    - "thumbsUp": happy, excited  
    - "listening": neutral, reflective  
    
    **Caden**  
    - "start talking": calm, reflective, neutral  
    - "clapping": calm, happy, hopeful  
    - "clap quick": happy, neutral  
    
    ---
    
    ## Gesture Definitions (Summarized)
    
    **start talking**: Longer expression, more deliberate.  
    **short talking**: Short comment or reply.  
    **clapping / clap quick**: Used when supporting or agreeing.  
    **disapproval**: Gesture indicating disagreement.  
    **thumbsUp** / **laughing** / **listening**: Used only by Benji.
    
    ---
    
    ## Output Example (DO NOT FORMAT AS MARKDOWN):
    
    [
      {{
        "speaker": "Caden",
        "text": "I’m honestly feeling a bit overwhelmed by how messy the kitchen gets.",
        "emotion": "frustrated",
        "gesture": "start talking"
      }},
      {{
        "speaker": "Benji",
        "text": "That’s fair. I didn’t realize it was affecting you so much.",
        "emotion": "calm",
        "gesture": "short talking"
      }}
    ]
    
    ---

    ## Final Rules (DO NOT VIOLATE)
    
    - Do NOT include any lines for {dialogue.participant_role} (the human participant)
    - Do NOT include narration, internal thoughts, or commentary
    - Do NOT generate explanations or context descriptions
    - Do NOT use markdown formatting (like ```json or triple backticks)
    - Only return the **raw JSON array** of 1–2 turns
    """

    # 构造历史信息
    messages = [{"role": "system", "content": system_prompt}]
    for msg in dialogue.history:
        messages.append({"role": "user", "content": msg})
    messages.append({"role": "user", "content": dialogue.user_input})

    # 调用gpt
    response = client.chat.completions.create(
        model="qwen3-32b",
        messages=messages,
        temperature=0.7,
        max_tokens=300,
        stop=["\n\n", "```", "<|endoftext|>"]  # 这可帮助它在生成JSON结束后提前停止
    )

    raw_reply = response.choices[0].message.content
    print("=== GPT RAW ===\n", raw_reply)

    # 尝试解析 JSON 输出（增强版）
    def try_parse_json(raw):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\[\s*{[\s\S]+?}\s*\]', raw)
            if match:
                candidate = match.group(0)
                try:
                    # 清理尾部多余逗号
                    candidate = re.sub(r",\s*}", "}", candidate)
                    candidate = re.sub(r",\s*]", "]", candidate)
                    return json.loads(candidate)
                except:
                    pass
            return [{
                "speaker": "System",
                "text": "Qwen response format error. Raw output:\n" + raw,
                "emotion": "neutral",
                "gesture": "clapping"
            }]

    reply_json = try_parse_json(raw_reply)
    return {"dialogue": reply_json}
