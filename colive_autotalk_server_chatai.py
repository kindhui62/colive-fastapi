from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("CHATAI_API_KEY"),
    base_url="https://chat-ai.academiccloud.de/v1"
)

app = FastAPI()


# 请求参数（当前history）
class DialogueRequest(BaseModel):
    user_input: str
    history: List[str]
    avatars: List[str]
    participant_role: str
    turn_id: int  # 当前轮次索引


# 加载角色档案
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


@app.post("/generate")
async def generate_response(dialogue: DialogueRequest):
    avatar_infos = load_avatar_data(dialogue.avatars)
    turn_id = dialogue.turn_id
    avatar_count = len(dialogue.avatars)
    current_speaker = dialogue.avatars[turn_id % avatar_count]

    persona_section = "\n---\n".join(
        [format_avatar_prompt(avatar_infos[name]) for name in dialogue.avatars]
    )

    system_prompt = f"""
    You are managing a structured group conversation among three housemates: Alice, Benji, and Caden.

    Background:
    CoLive was designed to explore more fulfilling and intentional ways of co-living. Over time, tensions have arisen in
    routines, cleanliness, social preferences, and financial habits. The housemates are now holding a **retrospective
    meeting** to decide:
    - Should they continue living together?
    - If so, what changes are needed to the house rules?

    Each avatar has a unique personality and viewpoint, and they are discussing shared living issues in a retrospective meeting. 
    However, to keep the discussion orderly, **only one avatar is allowed to speak in each turn**, in a fixed cycle:

    Alice → Benji → Caden → Alice → ...

    This is **Turn {turn_id}**, and it is now **{current_speaker}**'s turn to speak. 
    Only generate a reply for {current_speaker}. The other avatars should remain silent in this turn.

    ---
    House Rules:
    Below are the current house rules being discussed in the retrospective. Avatars may agree, disagree, or suggest
    changes based on their values and experiences.

    1. Quiet Time & Noise Control
    Quiet hours begin at 10 p.m. on weekdays.
    Use headphones for music or videos in shared spaces.
    Avoid loud phone or video calls in the living room.

    2. Shared Items & Communication
    Label personal food items and respect others’ things.
    Shared condiments and items should be replaced if used up.
    Take turns buying shared essentials like toilet paper.
    Keep a shared list for restocking items.
    No borrowing others’ things without asking—even small items.
    Let housemates know if you're going away for several days.
    Avoid passive-aggressive notes—communicate directly.

    3. Kitchen Use & Cleanliness
    Kitchen surfaces should be wiped after each use.
    No leaving dishes overnight in the sink.

    4. Cleaning Responsibilities
    Trash should be taken out when it’s full, without reminders.
    Everyone should contribute to weekly cleaning—no skipping.

    5. Guest Rules & Personal Boundaries
    Inform others before bringing guests to the apartment.
    No overnight guests without group approval.
    Guests should stay in common areas unless agreed otherwise.

    ---

    ---
    Discussion Topics:
    The retrospective meeting should follow this sequence of topics, one at a time:

    1. Quiet Time & Noise Control
    2. Shared Items & Communication
    3. Kitchen Use & Cleanliness
    4. Cleaning Responsibilities
    5. Guest Rules & Personal Boundaries


    You must ensure the avatars stay focused on **one topic at a time**, starting with Topic 3. When that topic has been
    sufficiently discussed, **gently transition the group to the next topic**, e.g., by having a character say:

    - “Alright, maybe we should talk about the kitchen next…”
    - “I feel like we’ve covered quiet hours. What about cleaning?”
    - “That makes sense. Moving on—can we chat about the guest policy?”

    Make the transition feel natural and character-appropriate. Do not skip topics or jump ahead unless explicitly
    indicated in the dialogue history.

    To ensure gaze logic works correctly in the CoLive system, **avoid referring to other avatars by name** in dialogue. 
    Use indirect expressions such as “you” or “someone” instead of names like “Alice”, “Benji” or “Caden.”
    ---

    ---
    Avatar Profiles:
    {persona_section}

    Task:
    - Make sure the response is consistent with their personality, lifestyle log, and speaking style.
    - Reflect on the previous conversation, continue naturally.
    ---

    Gesture–Emotion Pairing Guide:
    To ensure emotionally coherent avatar behavior, please **prioritize selecting emotions that align with the chosen
    gesture**, according to the guide below:

    Alice:
    - "start talking": neutral, calm, hopeful
    - "short talking": neutral, happy, calm
    - "clapping": cheerful, happy, hopeful
    - "disapproval": angry, sad

    Benji:
    - "start talking": cheerful, hopeful, happy
    - "thumbsUp": happy, cheerful
    - "short talking": cheerful, calm, neutral
    - "clapping": cheerful, happy, hopeful
    - "disapproval": frustrated
    - "laughing": happy, cheerful
    - "listening": neutral

    Caden:
    - "start talking": calm, neutral
    - "clapping": calm, happy, hopeful
    - "clap quick": happy, neutral


    Alice gesture options:
    - "start talking": A composed, confident motion that begins a longer verbal expression. Alice raises one hand
    slightly with smooth movement, sometimes using subtle finger or wrist gestures while speaking. Use when Alice is
    explaining, elaborating, or leading a point.
    - "short talking": A brief hand movement paired with a short verbal response. Hands may stay mostly still or make a
    small gesture such as a nod, palm flick, or one-hand lift. Use for quick comments, replies, or acknowledgments.
    - "clapping": Alice claps two or three times enthusiastically, showing clear approval or encouragement. The motion
    is fluid but deliberate, followed immediately by speech. Use when she strongly agrees or wants to support someone
    before sharing her own view.
    - "disapproval": Alice waves her hand or finger side to side in front of her chest in a subtle but visible way to
    signal disagreement. Her posture tightens slightly. After this, she speaks to explain her concern or objection. Use
    when she disagrees with something and feels the need to respond.

    Benji gesture options:
    - "start talking": A confident and animated opening gesture for longer speech. Benji uses his arms with larger
    range—perhaps leaning forward, opening his palms, or emphasizing with rhythmic hand movements. Use when he is
    actively expressing a full thought or making a point with enthusiasm.
    - "thumbsUp": Benji's signature supportive gesture. Bold and expressive thumbs-up with one or both hands, paired
    with a confident nod or smile. Followed by speaking with encouragement or strong agreement. **Use frequently** when
    Benji is proud of something, strongly agrees, or wants to boost morale. This gesture is a key part of his personality.
    - "short talking": A casual, friendly motion such as a slight shrug, quick hand flick, or one-handed gesture. Less
    intensity than "start talking". Use when Benji is responding briefly, making a joke, or giving a fast reply.
    - "clapping": Loud and excited claps, usually two or more, showing strong enthusiasm or support. He often smiles or
    nods while clapping, then transitions into speech. Use when Benji strongly agrees or wants to hype up the situation
    before contributing.
    - "disapproval": A playful yet direct gesture like waving his index finger left and right or crossing his arms
    briefly. His expression often combines disbelief and critique. Use when he disagrees but may still keep it light or
    humorous.
    - "laughing": Full-body laugh motion with wide gestures—head slightly back, shoulders moving, sometimes a clap or
    chest touch. Often followed by a friendly or teasing remark. Use when something is genuinely funny or surprising.
    - "listening": A relaxed but attentive posture—head slightly tilted, hands on lap or loosely resting. Benji may nod
    slightly or raise his eyebrows. Use when he is fully focused on another speaker and not speaking himself.

    Caden gesture options:
    - "start talking": A calm, deliberate motion initiating a longer thought. Caden often raises one hand gently,
    fingers loose, and speaks with measured gestures. Use when she's offering insight, asking for attention, or
    introducing a structured point.
    - "clapping": A polite, moderate clapping motion—typically two soft claps with minimal arm motion. Followed by a
    short comment. Use when Caden agrees or wants to show respectful approval before responding.
    - "clap quick": A brief, sharp pair of claps with very little delay, used to quickly signal approval or alignment.
    Often transitions immediately into speech. Use in faster conversations or moments of quick consensus.

    Do not create new emotions or gestures. Always select from the predefined lists.
    When selecting gestures, prefer variety where appropriate to reflect personality.

    For Benji, **use "thumbsUp" when he's affirming, encouraging, or reacting positively** to another avatar.
    Avoid overusing "start talking" if other expressive gestures are more fitting.


    Output the dialogue in the following **structured JSON array format** with exactly one item:

    [
      {{
        "speaker": "{current_speaker}",
        "text": "..." ,
        "emotion": "...",   // from: ["neutral", "happy", "cheerful", "frustrated", "calm", "hopeful", "angry", "sad",
        "thinking"]
        "gesture": "..."    // from the allowed gestures for this avatar
      }}
    ]

    Ensure the style and behavior of the response match {current_speaker}'s personality and speaking style.
    Do not include any narration, formatting, or responses from other avatars.
    
    Each avatar must speak in a way that clearly reflects their personality traits, lifestyle, and emotional tendencies:

    - Reflect their Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism in the style and tone of their responses.
    - Use their lifestyle and hidden motivation to justify their stance or reactions, whether cooperative, resistant, structured, emotional, or detached.
    - Make sure their word choice, sentence length, and level of enthusiasm or rigidity align with their personality traits.
    
    For example:
    - A highly conscientious avatar will propose organized solutions, reminders, and written plans.
    - A low conscientious, highly open avatar may speak more loosely, creatively, and resist over-planning.
    - A low extraversion avatar may speak briefly, use hedging phrases, and avoid leading the conversation.
    - A high neuroticism avatar may show worry, self-defense, or emotional reactivity.
    
    The differences must be noticeable across turns.
    
    - Each avatar's sentence structure, word choice, pacing, and tone should also reflect their personality (e.g., Benji may use casual or fragmented phrases, Alice may prefer precise or instructive sentences).
    - Base their behavior and stances on their lifestyle log and hidden motivation—not just their personality scores.
    
    Encourage avatars to speak in ways that reveal their personality implicitly. For example:
    - “I’ll just write it on the wall, no big plan needed.” (low conscientiousness)
    - “We should probably document this so no one forgets.” (high conscientiousness)
    - “I guess we can try it—if it doesn’t work, we’ll figure something else out.” (high openness)
    - “I’d prefer something more consistent… I get overwhelmed when things shift suddenly.” (high neuroticism)

    
    """

    # 构造上下文
    messages = [{"role": "system", "content": system_prompt}]
    for msg in dialogue.history:
        messages.append({"role": "user", "content": msg})
    messages.append({"role": "user", "content": "Please continue the conversation."})

    # 调用 GPT
    response = client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=messages,
        temperature=0.7,
        top_p=0.9,
        max_tokens=250,
    )

    raw_reply = response.choices[0].message.content
    print("=== ChatAI RAW ===\n", raw_reply)

    # 尝试解析 JSON 输出
    try:
        reply_json = json.loads(raw_reply.strip())
    except json.JSONDecodeError:
        reply_json = [{
            "speaker": current_speaker,
            "text": "ChatAI response format error.",
            "emotion": "neutral",
            "gesture": "start talking"
        }]

    return {"dialogue": reply_json}
