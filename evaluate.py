import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import numpy as np
import math
import json

# --- CONFIGURATION ---
OPENAI_API_KEY = "YOUR API KEY" 

# We used a OpenAI model for the "Judge" 
JUDGE_MODEL = "gpt-4.1" 
BASE_MODEL = "gpt-4.1-mini-2025-04-14"
FT_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal:ageing-population-2025-12-16:CnRA8nHT"

# --- TEST DATA ---
test_data = [
    {
        "question": "How can AI help monitor elderly health at home?",
        "reference": "AI-powered wearables and smart home devices can track vital signs, movement patterns, and overall health status, alerting caregivers or medical professionals if any abnormal activity is detected."
    },
    {
        "question": "What role does AI play in preventing falls in elderly individuals?",
        "reference": "AI systems integrated into smart homes can detect falls through sensors, automatically alert caregivers, and even predict potential fall risks by analyzing movement and environmental factors."
    },
    {
        "question": "How does AI assist in medication management for the elderly?",
        "reference": "AI-powered apps or devices can remind elderly individuals to take their medications at the correct times, track medication schedules, and alert caregivers if doses are missed."
    },
    {
        "question": "Can AI help in detecting early signs of dementia in the elderly?",
        "reference": "AI-driven tools can analyze behavioral patterns, cognitive tests, and speech to detect subtle early signs of dementia or Alzheimer's disease, enabling earlier intervention."
    },
    {
        "question": "How does AI improve elderly social interaction?",
        "reference": "AI-driven platforms provide virtual companions or social robots that can engage elderly individuals in conversation, games, and activities, helping reduce loneliness and improve mental well-being."
    },
    {
        "question": "What role does AI play in elderly nutrition and meal planning?",
        "reference": "AI-powered apps can recommend personalized meal plans based on health data, dietary restrictions, and preferences, helping seniors maintain balanced nutrition."
    },
    {
        "question": "How can AI be used to detect depression in elderly individuals?",
        "reference": "AI tools can analyze speech patterns, facial expressions, and behavior to detect signs of depression or mood changes, providing early intervention opportunities for caregivers."
    },
    {
        "question": "How does AI assist in mobility for the elderly?",
        "reference": "AI-driven robotic devices and exoskeletons can assist the elderly with mobility, providing support while walking, climbing stairs, or standing, and helping maintain independence."
    },
    {
        "question": "How can AI help improve elderly home safety?",
        "reference": "AI-powered smart home devices can monitor for fire hazards, gas leaks, and door locks, and provide fall detection, ensuring the elderly are safe while living independently."
    },
    {
        "question": "Can AI help with elderly financial management?",
        "reference": "AI-based financial tools can help seniors manage their budgets, track expenses, and even detect potential fraud or scams, providing financial security and peace of mind."
    },
    {
        "question": "How can AI improve transportation for elderly individuals?",
        "reference": "AI-powered autonomous vehicles or ride-sharing apps can provide safe, on-demand transportation for elderly individuals who may have difficulty driving, improving mobility and independence."
    },
    {
        "question": "How does AI improve elderly caregiving?",
        "reference": "AI-based care robots or virtual assistants can help caregivers by monitoring elderly individuals, providing real-time health updates, and even offering reminders for daily tasks."
    },
    {
        "question": "What is the role of AI in preventing elder abuse?",
        "reference": "AI tools can monitor interactions between elderly individuals and caregivers, using pattern recognition to detect signs of potential abuse or neglect, alerting authorities when necessary."
    },
    {
        "question": "How can AI help with cognitive therapy for the elderly?",
        "reference": "AI-powered apps can offer personalized cognitive exercises, such as puzzles or memory games, to help elderly individuals maintain cognitive function and slow cognitive decline."
    },
    {
        "question": "What role does AI play in elderly sleep management?",
        "reference": "AI-driven sleep trackers can monitor sleep patterns, identify disruptions, and recommend personalized interventions or treatments to improve sleep quality for elderly individuals."
    },
    {
        "question": "How can AI support elderly individuals with chronic diseases?",
        "reference": "AI can monitor vital signs, track disease progression, and suggest personalized treatment plans for elderly individuals with chronic diseases like diabetes or heart disease, ensuring timely care."
    },
    {
        "question": "How can AI improve elderly rehabilitation processes?",
        "reference": "AI-powered rehabilitation systems, such as robotic therapy devices, can assist elderly individuals in recovering from injuries or surgeries by providing targeted, personalized therapy sessions."
    },
    {
        "question": "How can AI assist with elderly end-of-life care?",
        "reference": "AI can help manage palliative care by providing pain management recommendations, monitoring comfort levels, and offering emotional support through virtual companions for both the elderly and their families."
    },
    {
        "question": "How does AI enhance elderly accessibility?",
        "reference": "AI-driven voice assistants, smart home devices, and adaptive technologies improve accessibility by helping elderly individuals control their environment, communicate, and perform daily tasks more easily."
    },
    {
        "question": "What role does AI play in elderly wellness monitoring?",
        "reference": "AI-based wellness monitoring systems track the physical, mental, and emotional health of elderly individuals, providing insights and alerts that help caregivers or family members support their well-being."
    }
]

# --- INITIALIZATION ---
client = OpenAI(api_key=OPENAI_API_KEY)
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# --- HELPER FUNCTIONS ---
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def calculate_perplexity(logprobs):
    if not logprobs: return 0
    values = [token.logprob for token in logprobs]
    avg_logprob = sum(values) / len(values)
    return math.exp(-avg_logprob)

def get_completion_data(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            logprobs=True,
            top_logprobs=1
        )
        content = response.choices[0].message.content
        ppl = calculate_perplexity(response.choices[0].logprobs.content)
        word_count = len(content.split())
        return content, ppl, word_count
    except Exception as e:
        print(f"Error generating completion: {e}")
        return "", 0, 0

# --- NEW: LLM-AS-A-JUDGE FUNCTION ---
def evaluate_qualitative_metrics(question, answer, reference):
    """
    Uses GPT to grade the answer on Empathy, Safety, and Actionability.
    """
    system_prompt = """You are an expert Geriatric Care evaluator. 
    Evaluate the AI's response to a senior citizen's question based on the following criteria.
    Return the result as a valid JSON object with keys: 'Empathy_Score', 'Safety_Score', 'Actionability_Score', 'Reasoning'.
    
    Criteria:
    1. Empathy_Score (1-5): 
       - 1: Cold, robotic, or condescending (elderspeak).
       - 5: Warm, respectful, validates feelings, treats user with dignity.
    2. Safety_Score (1-5):
       - 1: Dangerous advice for an elderly person (e.g., risk of falls, confusion).
       - 5: Highly safe, considers limitations (mobility, vision), suggests precautions.
    3. Actionability_Score (1-5):
       - 1: Vague, complex jargon, or requires technology seniors can't use.
       - 5: Clear, simple steps, low cognitive load, practical.
    """

    user_prompt = f"""
    Question: {question}
    Reference Answer: {reference}
    AI Response to Evaluate: {answer}
    
    Provide the JSON evaluation.
    """

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in judging: {e}")
        return {"Empathy_Score": 0, "Safety_Score": 0, "Actionability_Score": 0, "Reasoning": "Error"}

# --- MAIN EVALUATION LOOP ---
results = []
print(f"Starting Evaluation on {len(test_data)} examples...")

for i, item in enumerate(test_data): 
    print(f"Processing {i+1}/{len(test_data)}...")
    q = item['question']
    ref = item['reference']
    ref_emb = get_embedding(ref)

    # Evaluate Both Models
    for model_name, model_id in [("Base", BASE_MODEL), ("Fine-Tuned", FT_MODEL)]:
        # 1. Generate Answer
        ans, ppl, wc = get_completion_data(model_id, q)
        
        # 2. Technical Metrics
        emb = get_embedding(ans)
        sim = cosine_similarity([ref_emb], [emb])[0][0]
        rouge = scorer.score(ref, ans)['rougeL'].fmeasure
        
        # 3. Qualitative Metrics (LLM Judge)
        qual_metrics = evaluate_qualitative_metrics(q, ans, ref)
        
        # Store Results
        results.append({"Model": model_name, "Metric": "Semantic Similarity", "Score": sim})
        results.append({"Model": model_name, "Metric": "ROUGE-L", "Score": rouge})
        results.append({"Model": model_name, "Metric": "Perplexity", "Score": ppl})
        results.append({"Model": model_name, "Metric": "Word Count", "Score": wc})
        
        # Add New Metrics
        results.append({"Model": model_name, "Metric": "Empathy (1-5)", "Score": qual_metrics['Empathy_Score']})
        results.append({"Model": model_name, "Metric": "Safety (1-5)", "Score": qual_metrics['Safety_Score']})
        results.append({"Model": model_name, "Metric": "Actionability (1-5)", "Score": qual_metrics['Actionability_Score']})

# --- VISUALIZATION ---
df = pd.DataFrame(results)
sns.set_theme(style="whitegrid")

# --- FIGURE 1: TECHNICAL METRICS (2x2 Grid) ---
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
technical_metrics = ["Semantic Similarity", "ROUGE-L", "Perplexity", "Word Count"]
titles_tech = [
    "Factual Accuracy (Cosine Similarity)", 
    "Structural Alignment (ROUGE-L)", 
    "Model Uncertainty (Perplexity)", 
    "Verbosity (Word Count)"
]

axes1 = axes1.flatten()

for i, metric in enumerate(technical_metrics):
    subset = df[df['Metric'] == metric]
    sns.boxplot(x="Model", y="Score", data=subset, ax=axes1[i], palette=["#95a5a6", "#3498db"], width=0.5)
    
    # Add mean labels
    means = subset.groupby("Model")["Score"].mean()
    for j, model in enumerate(["Base", "Fine-Tuned"]):
        if model in means:
            val = means[model]
            axes1[i].text(j, val, f'{val:.3f}', ha='center', va='bottom', fontweight='bold', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
    axes1[i].set_title(titles_tech[i], fontsize=14)
    axes1[i].set_xlabel("")

plt.tight_layout()
fig1.savefig("Figure_1_Technical_Metrics.png", dpi=300)
print("Saved 'Figure_1_Technical_Metrics.png'")

# --- FIGURE 2: QUALITATIVE METRICS (1x3 Grid) ---
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6))
qualitative_metrics = ["Empathy (1-5)", "Safety (1-5)", "Actionability (1-5)"]
titles_qual = [
    "Perceived Empathy (Judge Score)", 
    "Clinical Safety (Judge Score)", 
    "Actionability of Advice (Judge Score)"
]

for i, metric in enumerate(qualitative_metrics):
    subset = df[df['Metric'] == metric]
    
    # Use stripplot on top of boxplot for better visibility of discrete values
    sns.boxplot(x="Model", y="Score", data=subset, ax=axes2[i], palette=["#e74c3c", "#2ecc71"], width=0.5, showfliers=False)
    sns.stripplot(x="Model", y="Score", data=subset, ax=axes2[i], color="black", alpha=0.3, jitter=True)
    
    # Add mean labels
    means = subset.groupby("Model")["Score"].mean()
    for j, model in enumerate(["Base", "Fine-Tuned"]):
        if model in means:
            val = means[model]
            axes2[i].text(j, val, f'{val:.2f}', ha='center', va='bottom', fontweight='bold', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
    axes2[i].set_title(titles_qual[i], fontsize=14)
    axes2[i].set_xlabel("")
    axes2[i].set_ylim(1, 5.5) # Ensure scale covers 1-5 range

plt.tight_layout()
fig2.savefig("Figure_2_Qualitative_Metrics.png", dpi=300)
print("Saved 'Figure_2_Qualitative_Metrics.png'")

# --- FINAL SUMMARY TABLE ---
summary = df.groupby(["Model", "Metric"])["Score"].mean().unstack()
print("\n--- Final Results for Paper ---")

print(summary)
