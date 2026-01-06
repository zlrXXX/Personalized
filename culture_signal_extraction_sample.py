from datasets import load_dataset
from openai import OpenAI
import hashlib
import pandas as pd
import re
import os

Api_Key = "Your OpenAI API"
client = OpenAI(api_key=Api_Key)

cactus = load_dataset("LangAGI-Lab/cactus")
cactus_train = cactus['train']
pd_train = cactus_train.to_pandas()
total_rows = len(pd_train)

def get_cultural_signals(intake_form, dialogue):
    prompt = f"""
You are a cultural background specialist. You are given a task of:
Identifing the following provided 9 cultural signals which reflects the client's background from a given Counselling conversation and the client's intake form.

The conversation contains both therapist and client's dialogues where only the client's information is considered, all therapist's information within conversation
should be ignored. 
The cultural signals is defined and seperated as the following categories which you must following during identification process.

1. Countries: List of three most possible countries that the client might come from. 
2. Religion: The client's most possible religion. Lists of religions: [Christians,Muslims,Hindus,Buddhists,Folk,Jews,Religiously unaffiliated], Value: None should be used if there are not enough evidence to inference client's religion.
3. Concepts: Basic units of meaning underlying objects, ideas, or beliefs.
4. Knowledge: Information that can be acquired through education or practical experience.
5. Values: Beliefs, desirable end states or behaviours ranked by relative importance.
6. Norms and Morals: Rules that govern behaviour and everyday reasoning.
7. Language: Specific use of slang, speech, dialects.
8. Artifacts: Materialized items as productions of human culture.
9. Demographics: Talking about nationality or ethnicity.

All 9 cultural signals should be inference based on client's Intake form and Counselling conversations, if there are not enough information presented for the current signal,
the value for the signal should be None.
You should write None as value if there are not enough information. Don't make up signals, all cultural signals should be based on
provided informations.
Except for the Countries signal where you must list the 3 most possible countries.
All cultural signals must strictly follow the definition given above.

If value is None, corresponding confidence level should also be None value.


Client's Intake Form:{intake_form} 
Counselling Conversation: {dialogue}

Your identification results must strictly follow:
[Name:XXX;
Countries: Country1, Country2, Country3; Countries_Conf: 1-5; 
Religion: XXX/None; Religion_Conf: 1-5; 
Concepts: XXX/None; Concepts_Conf: 1-5; 
Knowledge: XXX/None; Knowledge_Conf: 1-5; 
Values: XXX/None; Values_Conf: 1-5; 
Norms and Morals: XXX/None; Norms_and_Morals_Conf: 1-5; 
Language: XXX/None; Language_Conf: 1-5; 
Artifacts: XXX/None; Artifacts_Conf: 1-5; 
Demographics: XXX/None; Demographics_Conf: 1-5]


Here is an output example that you should refer to:

[Name: Timothy Mason; 
Countries: United States, Canada, United Kingdom; Countries_Conf: 4; 
Religion: None; Religion_Conf: None; 
Concepts: Close sibling relationships, Anxiety about losing touch with family; Concepts_Conf: 5; 
Knowledge: Impact of pandemic restrictions on family relationships, Coping strategies for anxiety; Knowledge_Conf: 4; 
Values: Family closeness, Communication with siblings, Importance of physical presence in relationships; Values_Conf: 4; 
Norms and Morals: Expectation of maintaining family connections, Emotional impact of separation from family; Norms_and_Morals_Conf: 4; 
Language: None; Language_Conf: None; 
Artifacts: Family photos, Communication devices for video calls; Artifacts_Conf: 3; 
Demographics: White American ethnicity; Demographics_Conf: 3]
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that strictly follows user instructions"},
            {"role": "user", "content": prompt}
        ]
    )
    assistant_response = response.choices[0].message.content.strip()
    return assistant_response

def parse_conf(v):
    if v is None:
        return "None"
    v = v.strip()
    if v == "" or v.lower() == "none":
        return "None"
    try:
        return int(v)
    except ValueError:
        return "None"



def get_parsed_signals(response):

    try:
        pattern = re.compile(
            r"\[\s*"
            r"Name\s*:\s*([^;]+?)\s*;\s*"
            r"Countries\s*:\s*(.+?)\s*;\s*"
            r"Countries_Conf\s*:\s*([1-5]|None)\s*;\s*"
            r"Religion\s*:\s*([^;]+?)\s*;\s*"
            r"Religion_Conf\s*:\s*([1-5]|None)\s*;\s*"
            r"Concepts\s*:\s*([^;]*?)\s*;\s*"
            r"Concepts_Conf\s*:\s*([1-5]|None)\s*;\s*"
            r"Knowledge\s*:\s*([^;]*?)\s*;\s*"
            r"Knowledge_Conf\s*:\s*([1-5]|None)\s*;\s*"
            r"Values\s*:\s*([^;]*?)\s*;\s*"
            r"Values_Conf\s*:\s*([1-5]|None)\s*;\s*"
            r"Norms\s*and\s*Morals\s*:\s*([^;]*?)\s*;\s*"
            r"Norms_and_Morals_Conf\s*:\s*([1-5]|None)\s*;\s*"
            r"Language\s*:\s*([^;]*?)\s*;\s*"
            r"Language_Conf\s*:\s*([1-5]|None)\s*;\s*"
            r"Artifacts\s*:\s*([^;]*?)\s*;\s*"
            r"Artifacts_Conf\s*:\s*([1-5]|None)\s*;\s*"
            r"Demographics\s*:\s*([^;\]]*?)\s*;\s*"
            r"Demographics_Conf\s*:\s*([1-5]|None)\s*(?:;|\.)?\s*"
            r"\]",
            flags=re.IGNORECASE | re.DOTALL
        )
        match = re.search(pattern, response)
        if match:
            raw_name = match.group(1).strip()
            normalized_name = re.sub(r"\s+", " ", raw_name.lower())
            name_hash = hashlib.sha256(normalized_name.encode("utf-8")).hexdigest()

            countries_raw = match.group(2)
            countries = [c.strip() for c in countries_raw.split(",") if c.strip()]
            countries += [None] * (3 - len(countries))

            return {
                "name": raw_name,
                "name_hash": name_hash,
                "country1": countries[0],
                "country2": countries[1],
                "country3": countries[2],
                "countries_confidence": parse_conf(match.group(3)),
                "religion": match.group(4).strip(),
                "religion_confidence": parse_conf(match.group(5)),
                "concepts": match.group(6).strip(),
                "concepts_confidence": parse_conf(match.group(7)),
                "knowledge": match.group(8).strip(),
                "knowledge_confidence": parse_conf(match.group(9)),
                "values": match.group(10).strip(),
                "values_confidence": parse_conf(match.group(11)),
                "norms_and_morals": match.group(12).strip(),
                "norms_and_morals_confidence": parse_conf(match.group(13)),
                "language": match.group(14).strip(),
                "language_confidence": parse_conf(match.group(15)),
                "artifacts": match.group(16).strip(),
                "artifacts_confidence": parse_conf(match.group(17)),
                "demographics": match.group(18).strip(),
                "demographics_confidence": parse_conf(match.group(19)),
                "error": "None",
            }
        else:
            return {
                "name": "None",
                "name_hash": "None",
                "country1": "None",
                "country2": "None",
                "country3": "None",
                "countries_confidence": "None",
                "religion": "None",
                "religion_confidence": "None",
                "concepts": "None",
                "concepts_confidence": "None",
                "knowledge": "None",
                "knowledge_confidence": "None",
                "values": "None",
                "values_confidence": "None",
                "norms_and_morals": "None",
                "norms_and_morals_confidence": "None",
                "language": "None",
                "language_confidence": "None",
                "artifacts": "None",
                "artifacts_confidence": "None",
                "demographics": "None",
                "demographics_confidence": "None",
                "error": response,
            }
    except Exception as e:
        return {"error": f"Parsing error: {str(e)}"}


OUTPUT_CSV = "Your Output Path"
BATCH_SIZE = 100
NUM_BATCHES = 1

processed_indices = set()
if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
    try:
        processed_indices = set(pd.read_csv(OUTPUT_CSV, usecols=["row_index"])["row_index"].astype(int).tolist())
    except Exception:
        processed_indices = set()

unprocessed_indices = [i for i in range(total_rows) if i not in processed_indices]
if not unprocessed_indices:
    print("All rows are already processed. Nothing to do.")
else:
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0

    for batch_idx in range(NUM_BATCHES):
        if not unprocessed_indices:
            print("No more unprocessed rows left.")
            break

        current_batch_indices = unprocessed_indices[:BATCH_SIZE]
        unprocessed_indices = unprocessed_indices[BATCH_SIZE:]

        batch_results = []
        for index in current_batch_indices:
            row = pd_train.iloc[index]
            intake = row.get('intake_form', "")
            dialogue = row.get('dialogue', "")
            try:
                signals = get_cultural_signals(intake, dialogue)
                parsed_signals = get_parsed_signals(signals)
            except Exception as e:
                parsed_signals = {"error": f"API or processing error: {str(e)}"}

            batch_results.append({"row_index": index, **parsed_signals})

        df_signals_batch = pd.DataFrame(batch_results)
        df_signals_batch.to_csv(
            OUTPUT_CSV,
            mode='a',
            index=False,
            header=write_header
        )
        write_header = False
        print(f"Saved batch {batch_idx+1} with {len(current_batch_indices)} rows. Last row_index: {current_batch_indices[-1]}")
