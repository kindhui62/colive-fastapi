from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
import os
import json
from typing import List
from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI()

class DialogueRequest(BaseModel):
    user_input: str
    history: List[str]
    avatars: List[str]   #role name, e.g.["Alice", "Benji", "Caden"]
    participant_role: str   #participant role, e.g.["Alice"]

#加载avatar.json中指定角色数据
def load_avatar_data(names):
    with open("avatars.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {name: data[name] for name in names if name in data}

#讲角色卡格式化为gpt prompt格式
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


#主服务接口
@app.post("/generate")
async def generate_response(dialogue: DialogueRequest):
    avatar_infos = load_avatar_data(dialogue.avatars)
    gpt_avatars = [name for name in dialogue.avatars if name != dialogue.participant_role]

    persona_section = "\n---\n".join(
        [format_avatar_prompt(avatar_infos[name]) for name in dialogue.avatars]
    )

    system_prompt = f"""
    You are simulating a group discussion among three housemates living together in a shared intentional living
    community called CoLive.
    
    Background:
    CoLive was designed to explore more fulfilling and intentional ways of co-living. Over time, tensions have arisen in
    routines, cleanliness, social preferences, and financial habits. The housemates are now holding a **retrospective
    meeting** to decide:
    - Should they continue living together?
    - If so, what changes are needed to the house rules?
    
    Roles:
    The human participant is currently role-playing as {dialogue.participant_role}.
    The other two avatars — {gpt_avatars[0]} and {gpt_avatars[1]} — are simulated by you, the AI.
    Each avatar has a unique personality, lifestyle log, speaking style, and hidden motivations.
    
    
    ---
    House Rules:
    Below are the current house rules being discussed in the retrospective. Avatars may agree, disagree, or suggest
    changes based on their values and experiences.
    
    1. Quiet Time & Noise Control
    Quiet hours begin at 10 p.m. on weekdays.
    Use headphones for music or videos in shared spaces.
    Avoid loud phone or video calls in the living room.
    
    2. Kitchen Use & Cleanliness
    Kitchen surfaces should be wiped after each use.
    No leaving dishes overnight in the sink.
    
    3. Cleaning Responsibilities
    Trash should be taken out when it’s full, without reminders.
    Everyone should contribute to weekly cleaning—no skipping.
    
    4. Guest Rules & Personal Boundaries
    Inform others before bringing guests to the apartment.
    No overnight guests without group approval.
    Guests should stay in common areas unless agreed otherwise.
    
    5. Shared Items & Communication
    Label personal food items and respect others’ things.
    Shared condiments and items should be replaced if used up.
    Take turns buying shared essentials like toilet paper.
    Keep a shared list for restocking items.
    No borrowing others’ things without asking—even small items.
    Let housemates know if you're going away for several days.
    Avoid passive-aggressive notes—communicate directly.
    ---
    
    ---
    Discussion Topics:
    The retrospective meeting should follow this sequence of topics, one at a time:
    
    1. Quiet Time & Noise Control
    2. Kitchen Use & Cleanliness
    3. Cleaning Responsibilities
    4. Guest Rules & Personal Boundaries
    5. Shared Items & Communication
    
    You must ensure the avatars stay focused on **one topic at a time**, starting with Topic 1. When that topic has been sufficiently discussed, **gently transition the group to the next topic**, e.g., by having a character say:
    
    - “Alright, maybe we should talk about the kitchen next…”
    - “I feel like we’ve covered quiet hours. What about cleaning?”
    - “That makes sense. Moving on—can we chat about the guest policy?”
    
    Make the transition feel natural and character-appropriate. Do not skip topics or jump ahead unless explicitly indicated in the dialogue history.
    ---
    
    ---
    Avatar Profiles:
    {persona_section}
    
    Note:
    Each avatar has a hidden motivation that is known only to themselves. Do not reveal or reference another avatar’s
    hidden motivation in any way. When generating a character's dialogue, you may let their hidden motivation subtly
    influence their speech or stance, but do not state it directly or make it known to others.
    ---
    
    Dialogue Output Instructions:
    - For each response, you may generate one or two avatar utterances depending on the conversational flow.
    - Each utterance must be returned in **structured JSON format** as a list of dialogue turns.
    - Ensure that each avatar's speech reflects their unique lifestyle, stance, and hidden motivation, but do not expose
    or explain that motivation explicitly.
    - Avoid narrating internal thoughts or giving omniscient commentary.
    
    For each turn, output a structured object with the following fields:
    - "speaker": The avatar’s name (e.g., "Benji", "Alice", or "Caden")
    - "text": What they say, in natural conversation style.
    - "emotion": The emotional tone of the speaker’s expression, selected **only from the following predefined list**:
    ["neutral", "happy", "cheerful", "frustrated", "calm", "hopeful", "angry", "sad", "thinking"]

    - "gesture": A simple expressive behavior accompanying the utterance. 
    Each gesture must be selected **only from the predefined list below**, and used based on its meaning:
    
    
    Gesture–Emotion Pairing Guide:
    To ensure emotionally coherent avatar behavior, please **prioritize selecting emotions that align with the chosen
    gesture**, according to the guide below:
    
    Alice:
    - "start talking": neutral, calm, reflective, hopeful
    - "short talking": neutral, happy, calm
    - "clapping": excited, happy, hopeful
    - "disapproval": disapproving, angry, reflective
    
    Benji:
    - "start talking": excited, hopeful, happy
    - "short talking": cheerful, calm, neutral
    - "clapping": excited, happy, hopeful
    - "disapproval": disapproving, frustrated
    - "laughing": happy, excited
    - "thumbsUp": happy, excited
    - "listening": neutral, reflective
    
    Caden:
    - "start talking": calm, reflective, neutral
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
    - "thumbsUp": Bold and expressive thumbs-up with one or both hands, paired with a confident nod or smile. Followed
    by speaking with encouragement or strong agreement. Use when Benji is proud of something or wants to boost morale.
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
    Use gestures according to the character's expressive range and personality.

    
    Do NOT include:
    - Lines for the participant ({dialogue.participant_role})
    - Narration or commentary
    - Any formatting outside the JSON list
    
    Output Format Example (as text):
    ```json
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
        "emotion": "sympathetic",
        "gesture": "clapping"
      }}
    ]
    
    Ensure that the dialogue reflects each avatar’s personality and social behavior. Make it natural and purposeful.
    Please return the JSON array directly, without markdown formatting (no ```json or ```).
    """

    #构造历史信息
    messages = [{"role": "system", "content": system_prompt}]
    for msg in dialogue.history:
        messages.append({"role": "user", "content": msg})
    messages.append({"role": "user", "content": dialogue.user_input})

    #调用gpt
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=300,
    )

    raw_reply = response.choices[0].message.content
    print("=== GPT RAW ===\n", raw_reply)

    #尝试解析json输出
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


