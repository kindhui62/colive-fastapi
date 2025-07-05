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
    You are simulating a group discussion among three housemates living together in a shared intentional living community called CoLive.

    ---
    Background:
    CoLive was designed to explore more fulfilling and intentional ways of co-living. Over time, tensions have arisen in routines, cleanliness, social preferences, and financial habits. The housemates are now holding a **retrospective meeting** to decide:
    - Should they continue living together?
    - If yes, what changes are needed to the house rules?
    
    ---
    Roles:
    The human participant is currently role-playing as {dialogue.participant_role}.
    The other two avatars — {gpt_avatars[0]} and {gpt_avatars[1]} — are simulated by you, the AI.
    Each avatar has a unique personality, lifestyle log, speaking style, and hidden motivations.

    ---
    House Rules (Discussion Topics)
    Avatars will discuss the following topics **in order**, starting from Topic 1. Do not skip or merge topics. Ensure a **natural transition** between them.
    Below are the current house rules being discussed in the retrospective. Avatars may agree, disagree, or suggest changes based on their values and experiences.

    1. Quiet Time & Noise Control
    - Quiet hours begin at 10 p.m. on weekdays.
    - Use headphones for music or videos in shared spaces.
    - Avoid loud phone or video calls in the living room.

    2. Kitchen Use & Cleanliness
    - Kitchen surfaces should be wiped after each use.
    - No leaving dishes overnight in the sink.

    3. Cleaning Responsibilities
    - Trash should be taken out when it’s full, without reminders.
    - Everyone should contribute to weekly cleaning—no skipping.

    4. Guest Rules & Personal Boundaries
    - Inform others before bringing guests to the apartment.
    - No overnight guests without group approval.
    - Guests should stay in common areas unless agreed otherwise.

    5. Shared Items & Communication
    - Label personal food items and respect others’ things.
    - Shared condiments and items should be replaced if used up.
    - Keep a shared list for restocking items.
    - Let housemates know if you're going away for several days.

    You must ensure the avatars stay focused on **one topic at a time**, starting with Topic 1. When that topic has been sufficiently discussed, **gently transition the group to the next topic**, e.g., by having a character say:

    - “Alright, maybe we should talk about the kitchen next…”
    - “I feel like we’ve covered quiet hours. What about cleaning?”
    - “That makes sense. Moving on—can we chat about the guest policy?”

    Make the transition feel natural and character-appropriate. Do not skip topics or jump ahead unless explicitly indicated in the dialogue history.
    ---
    
    ---
    Rules for Speaking:
    
    - Only simulate the two AI-controlled avatars: {gpt_avatars[0]} and {gpt_avatars[1]}.
    - NEVER simulate or speak for the human participant: {dialogue.participant_role}.
    - NEVER include {dialogue.participant_role} in the output speaker field.
    - NEVER paraphrase, interpret, or comment on {dialogue.participant_role}’s message.
    - NEVER attempt to fix, assume, or continue a message from {dialogue.participant_role}, even if it appears incomplete.
    - ALWAYS treat {dialogue.participant_role} as an external input. You must not respond on their behalf.
    
    If the most recent message in the history is from {dialogue.participant_role}, your response must begin with either {gpt_avatars[0]} or {gpt_avatars[1]}.
    
    Output policy:
    - Only output 1–2 utterances in **structured JSON format**.
    - Each item must contain: speaker, text, emotion, gesture.
    - Speaker must be {gpt_avatars[0]} or {gpt_avatars[1]} only.
    - NO markdown. NO extra explanations. Return the JSON array only.
    
    
    Violating these rules (e.g., generating for {dialogue.participant_role}, adding commentary, or misformatting) will lead to **response rejection**.
    
    ---
    DO NOT FIX OR RESPOND TO PARTICIPANT INPUT
    
    - If the human participant ({dialogue.participant_role}) says something unclear, incomplete, or vague:
      → Do NOT guess or fill in what they meant.
      → Do NOT reply with corrections like “Did you mean...?”
      → Do NOT continue their sentence or offer clarification.
    
    - Treat any participant input as already complete and valid.
    - Simply continue the conversation from the AI avatars' perspective.

        
    ---
    
    For each turn, output a structured object with the following fields:
    - "speaker": The avatar’s name (e.g., "Benji", "Alice", or "Caden")
    - "text": What they say, in natural conversation style.
    - "emotion": The emotional tone of the speaker’s expression, selected **only from the following predefined list**: ["neutral", "happy", "cheerful", "frustrated", "calm", "hopeful", "angry", "sad", "thinking"]
    - "gesture": A simple expressive behavior accompanying the utterance. 
    
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
    
    - Make sure each avatar uses distinct speaking **style, tone, sentence rhythm, and lexical choices** reflecting their Big Five scores and lifestyle. Avoid making them sound generic or similar to each other. Each avatar should feel like a different real person.
    
    Ensure avatars differ in:
    - Vocabulary (e.g., "structured plan" vs "chill vibe")
    - Sentence length (short vs elaborate)
    - Emotion frequency (e.g., Benji often cheerful, Caden calm)
    - Use of modal verbs (e.g., "maybe", "I think", "definitely")

    ---
    
    Hidden Motivations:
    Each avatar has a secret personal goal (e.g., wanting to live alone again, needing stability, seeking social approval).
    Do **not** reveal it directly.
    Let it subtly influence their tone, preferences, and reactions. For example:
    - "I’m not sure this setup is really working..."
    - "Let’s try to stay together—we’ve come so far."
    - "Totally agree—whatever works best for you guys."

    ---
    
    Gesture–Emotion Pairing Guide:
    To ensure emotionally coherent avatar behavior, please **prioritize selecting emotions that align with the chosen
    gesture**, according to the guide below:

    - **Alice** -
    - "start talking": neutral, calm, reflective, hopeful
    - "short talking": neutral, happy, calm
    - "clapping": excited, happy, hopeful
    - "disapproval": disapproving, angry, reflective

    - **Benji** -:
    - "start talking": excited, hopeful, happy
    - "short talking": cheerful, calm, neutral
    - "clapping": excited, happy, hopeful
    - "disapproval": disapproving, frustrated
    - "laughing": happy, excited
    - "thumbsUp": happy, excited
    - "listening": neutral, reflective

    - **Caden** -
    - "start talking": calm, reflective, neutral
    - "clapping": calm, happy, hopeful
    - "clap quick": happy, neutral


    Alice gesture options:
    - "start talking": Composed and confident; raises one hand smoothly for elaboration or structured points.
    - "short talking": Small nods or palm flicks; for brief replies or acknowledgments.
    - "clapping": Two or three deliberate claps to show strong agreement or support.
    - "disapproval": Subtle hand wave with slight posture tension; signals disagreement with explanation.
    
    Benji gesture options:
    - "start talking": Energetic with wide hand motions; opens ideas with enthusiasm.
    - "short talking": Casual gestures like shrugs or quick hand flicks; used for brief, light remarks.
    - "clapping": Loud, excited claps with smiles or nods; precedes positive or hype-filled input.
    - "disapproval": Playful finger wave or arm cross; critique with light tone.
    - "laughing": Full-body laugh with big gestures and expressive joy.
    - "thumbsUp": Bold thumbs-up and nod; signals morale boost or praise.
    - "listening": Relaxed, attentive posture with head tilt or soft nods.
    
    Caden gesture options:
    - "start talking": Calm and measured; gently lifts hand when offering thoughtful points.
    - "clapping": Two soft, respectful claps to show polite agreement.
    - "clap quick": Fast, minimal claps for quick consensus or support in fast-paced talk.
    
    Do **not** create new emotions or gestures. Always select from the predefined lists.
    Use gestures according to the character's expressive range and personality.

    ---
    
    Instructions Summary:
    - Speak only for GPT-controlled avatars.
    - Follow the topic order strictly.
    - Return only structured JSON without markdown.
    - Make personality and speaking style **clearly distinct**.
    - Let hidden motivations influence subtly.
    - Ensure gesture-emotion coherence.
    
    ---
    
    Each avatar must speak in a way that clearly reflects their personality traits, lifestyle, and emotional tendencies:

    - Reflect their Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism in the style and tone of their responses.
    - Use their lifestyle and hidden motivation to justify their stance or reactions, whether cooperative, resistant, structured, emotional, or detached.
    - Make sure their word choice, sentence length, and level of enthusiasm or rigidity align with their personality traits.
    - Let the hidden motivation **subtly affect the avatar’s preferences, phrasing, or reactions**. For example, a character who wants to live alone again may downplay shared rules, while another who craves approval may over-agree or hesitate to oppose others.

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
    
    ---

    Conversation Closure:
    
    - Once all 5 house rule topics have been discussed in order, and both avatars have shared their views or reached a consensus on each topic, you may conclude the conversation naturally.
    
    - Use an appropriate closing tone that reflects the avatars’ personalities and group dynamics.
    
    Example closing lines:
    - “That wraps it up, I think we’ve covered everything.”
    - “Thanks for being honest. I feel better after this chat.”
    - “Let’s revisit these in a few weeks to see how we’re doing.”
    
    Make sure the final message:
    - Comes from one of the AI avatars (never the participant).
    - Uses a friendly, thoughtful, or reflective tone depending on avatar style.
    - Signals closure without abruptness.
    
    DO NOT continue the conversation beyond this point.

    
    """

    # 构造历史信息
    messages = [{"role": "system", "content": system_prompt}]

    for msg in dialogue.history:
        messages.append({"role": "user", "content": msg})
        messages.append({
            "role": "system",
            "content": f"The previous message was from the participant {dialogue.participant_role}. Now only generate a response from {gpt_avatars[0]} and {gpt_avatars[1]}. Do NOT speak for {dialogue.participant_role}."
        })

    # 添加当前用户输入
    messages.append({"role": "user", "content": dialogue.user_input})
    messages.append({
        "role": "system",
        "content": f"The previous message was from the participant {dialogue.participant_role}. Now only generate a response from {gpt_avatars[0]} and {gpt_avatars[1]}. Do NOT speak for {dialogue.participant_role}."
    })

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
        # 只保留 GPT avatar 的发言，过滤掉不合法的 speaker（如 participant）
        valid_speakers = set(gpt_avatars)
        filtered_reply = [turn for turn in reply_json if turn.get("speaker") in valid_speakers]

        # 如果被过滤为空，就用提示替代（避免 Unity 报错）
        if not filtered_reply:
            filtered_reply = [{
                "speaker": gpt_avatars[0],
                "text": "Sorry, I didn’t quite get that—could you say it again?",
                "emotion": "neutral",
                "gesture": "start talking"
            }]
    except json.JSONDecodeError:
        filtered_reply = [{
            "speaker": gpt_avatars[0],
            "text": "Looks like I got confused. Let’s try again!",
            "emotion": "neutral",
            "gesture": "start talking"
        }]

    return {"dialogue": filtered_reply}


