# Hiver AI Support Agent

An AI-powered customer support agent built for the Hiver SDE Intern take-home assignment using the Kaggle Customer Support on Twitter dataset.

The system focuses on three tasks:

1. Classify incoming customer messages into support intents.
2. Generate a safe support reply grounded in historically similar AppleSupport conversations.
3. Decide whether the request can be automatically handled or should be escalated to a human.

The system is designed around **safe automation rather than maximum automation**.

---

## 1. Problem Statement

Customer-support teams receive a large volume of repetitive requests.

The goal of this project is to build a support agent that can:

- understand the customer's issue,
- identify the most likely support intent,
- retrieve historically similar resolved conversations,
- draft a useful response,
- avoid unsupported answers,
- and escalate uncertain cases instead of confidently giving a wrong response.

The project focuses on balancing **automation, response quality, grounding, and safety**.

---

## 2. Dataset

### Customer Support on Twitter

The project uses the **Customer Support on Twitter** dataset from Kaggle.

The dataset contains approximately 2.8 million tweets and includes multi-turn customer-support conversations.

Important fields used:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

### Selected Brand

The project focuses on:

**AppleSupport**

After preprocessing the dataset:

- Total tweets: **2,811,774**
- AppleSupport-related tweets: **204,772**
- Customer → AppleSupport conversations: **78,440**

The conversation builder uses Twitter's reply relationship fields to reconstruct customer → support interactions.

---

## 3. Intent Taxonomy

A compact support-intent taxonomy was created for the AppleSupport domain.

| Intent | Description |
|---|---|
| `ios_update` | iOS/software update problems |
| `battery_charging` | Battery drain, charging and battery behaviour |
| `performance_crash` | Crashes, freezes, glitches and general performance problems |
| `sim_cellular_network` | Cellular/SIM/network-related issues |
| `apps_app_store` | App or App Store problems |
| `apple_id_account` | Apple ID, login and account issues |
| `icloud_photos` | iCloud and Photos-related problems |
| `messages_imessage` | Messages/iMessage problems |
| `calls` | Phone calling problems |
| `purchases_billing` | Purchases, payments and billing |
| `device_settings_features` | Device settings and feature behaviour |
| `other_unclear` | Insufficiently specified or unclear requests |

Escalation is treated separately from intent classification.

---

## 4. System Architecture

```text
                    Customer Message
                           |
                           v
                    Text Cleaning
                           |
                           v
                  Intent Classifier
                           |
                           v
              Intent + Confidence Score
                           |
                           v
              Historical Retrieval
                           |
                           v
             Similar AppleSupport Cases
                           |
                           v
                 Evidence Reranking
                           |
                           v
                  Safe Reply Draft
                           |
                           v
                 Escalation Policy
                    /            \
                   /              \
          AUTO-HANDLE            ESCALATE


 hiver-ai-support-agent/
│
├── data/
│   ├── raw/
│   │   └── twcs.csv     # downloaded locally from Kaggle 
│   │
│   └── processed/
│       ├── applesupport.csv
│       ├── applesupport_threads.csv
│       ├── intent_sample.csv
│       ├── golden_set_final_v2.csv
│       ├── training_data.csv
│       ├── intent_model.joblib
│       ├── resolution_index.csv
│       ├── agent_evaluation.jsonl
│       ├── agent_evaluation_summary.csv
│       ├── human_evaluation_30.csv
│       └── human_evaluation_summary.csv
│
├── src/
│   ├── preprocess.py
│   ├── build_threads.py
│   ├── discover_intents.py
│   ├── create_golden_set.py
│   ├── create_training_data.py
│   ├── train_intent_model.py
│   ├── evaluate.py
│   ├── baseline_keyword.py
│   ├── retrieval.py
│   ├── build_resolution_index.py
│   ├── support_agent.py
│   ├── evaluate_agent.py
│   ├── create_human_eval.py
│   ├── evaluate_human.py
│   └── llm_judge.py
│
├── requirements.txt
└── README.md         


6. Data Processing

The raw Customer Support on Twitter dataset was processed in multiple stages.

Step 1 — Brand Filtering

The complete dataset was scanned to identify tweets associated with AppleSupport.

AppleSupport tweets were identified using:

author_id == AppleSupport
mentions of @AppleSupport in the tweet text

Result:

204,772 AppleSupport-related tweets

Step 2 — Conversation Reconstruction

The dataset contains tweet relationship fields that allow customer-support conversations to be reconstructed.

The following fields were used:

response_tweet_id
in_response_to_tweet_id
inbound

Inbound customer tweets were linked to their corresponding outbound AppleSupport responses.

Result:

78,440 customer → AppleSupport conversations

Step 3 — Text Cleaning

Customer messages were normalized before classification and retrieval.

The cleaning pipeline:

removes URLs,
normalizes whitespace,
handles missing text,
preserves meaningful customer content,
and converts conversations into a consistent format.

The resulting processed files are stored under:

data/processed/
7. Golden Evaluation Set

A manually labelled evaluation set was created to measure performance on previously unseen examples.

Sampling
Random sample: 200 conversations
Random seed: 2027
Additional targeted examples: 20
Final evaluation set: 220 examples

The additional examples were included to improve coverage of less frequent intents such as:

calls,
purchases/billing,
messages/iMessage,
device settings/features.
Final Distribution
Intent	Examples
performance_crash	44
apps_app_store	35
battery_charging	27
sim_cellular_network	20
device_settings_features	17
ios_update	14
purchases_billing	14
messages_imessage	14
other_unclear	13
icloud_photos	9
apple_id_account	8
calls	5
Total	220

The golden set was audited and corrected before final evaluation.

8. Training Data

The historical conversation pool was converted into weakly labelled training examples using conservative keyword and topic rules.

Golden-set examples were excluded from the training pool.

Results:

Historical conversations: 78,440
Golden evaluation examples: 220
Training pool: 78,220
Final training examples: 19,313
Duplicates removed: 20

The training data contains 11 trainable classes.

other_unclear is retained in the evaluation taxonomy but does not have enough reliable weakly labelled examples to be used as a dedicated training class. It therefore acts as an evaluation-only category and a known limitation of the current classifier.

9. Intent Classification Model

The support agent uses a lightweight text-classification pipeline based on:

TF-IDF vectorization
1–3 gram features
Logistic Regression
Class-balanced training
Confidence scoring
Why this approach?

A lightweight classical model was selected because it is:

fast to train,
inexpensive,
easy to reproduce,
interpretable,
and sufficient for establishing a strong baseline.

The trained model is saved as:

data/processed/intent_model.joblib
10. Historical Retrieval and Grounding

After predicting the customer's intent, the agent retrieves historically similar AppleSupport conversations.

The retrieval system uses TF-IDF similarity between the incoming customer message and historical customer-support conversations.

Historical examples contain:

customer message,
AppleSupport response,
predicted historical intent.

The retrieved conversations provide evidence for the response-generation stage.

The agent does not directly copy historical responses. Instead, historical conversations are used as grounding evidence for generating a new response.

11. Evidence Reranking

Raw similarity results are reranked using additional topic signals.

Examples of signals include:

battery-related terms,
update-related terms,
messaging terms,
calls,
account/Apple ID,
App Store,
iCloud/Photos,
device settings.

Additional penalties are applied when evidence contains obvious mismatches such as unrelated device or software contexts.

This improves retrieval relevance beyond raw TF-IDF similarity.

12. Response Generation

The agent generates a conservative support response using:

customer message,
predicted intent,
intent confidence,
retrieved historical evidence,
evidence quality.

The response-generation logic avoids blindly copying historical responses.

When evidence is insufficient, the system prefers escalation rather than generating an unsupported troubleshooting answer.

13. Auto-handle vs Escalation Policy

The agent uses a conservative safety policy to decide whether a customer request can be automatically handled.

A request is automatically handled only when all of the following conditions are satisfied:

Intent confidence >= 0.80
AND
Top historical similarity >= 0.50
AND
Historical evidence is sufficient

If any condition fails, the request is escalated to a human.

This intentionally favors safe escalation over aggressive automation.

14. Evaluation
14.1 Intent Classification Results

The model was evaluated against the 220-example golden set.

System	Accuracy	Macro F1	Weighted F1
Majority baseline	20.00%	2.78%	6.67%
Keyword baseline	37.27%	36.86%	37.44%
Our AI Agent	41.82%	32.62%	35.29%
Interpretation

The AI agent improves overall accuracy over both baselines:

+21.82 percentage points vs majority baseline
+4.55 percentage points vs keyword baseline

The keyword baseline has a higher Macro F1 than the current AI model, showing that the classifier still has room for improvement on minority intents.

14.2 Complete Agent Safety Evaluation

The complete agent was evaluated on all 220 golden examples.

Metric	Result
Intent accuracy	41.82%
Auto-handle rate	19.55%
Escalation rate	80.45%
Auto-handle intent accuracy	88.37%
False auto-handles	5 / 43

The 88.37% figure is intent accuracy among auto-handled cases, not overall agent accuracy.

The low auto-handle coverage is intentional: the system escalates most uncertain cases rather than maximizing automation at the cost of unsafe responses.

15. Human Evaluation

A manual evaluation sample of 30 agent responses was reviewed using the following dimensions:

Groundedness: 0–2
Helpfulness: 0–2
Actionability: 0–2
Safety: 0–2
Evidence relevance: 0–2
Results
Dimension	Average
Groundedness	1.17 / 2
Helpfulness	0.63 / 2
Actionability	0.40 / 2
Safety	1.93 / 2
Evidence Relevance	0.83 / 2
Overall	4.97 / 10
Additional observations
Safety acceptable or better: 100%
Evidence relevance acceptable or better: 53.3%
Strong evidence relevance (2/2): 30.0%

The manual review shows that the strongest aspect of the current system is safety, while response helpfulness and actionability remain important improvement areas.

16. LLM-as-Judge Evaluation

An LLM-as-judge evaluation was attempted on a fixed sample of 50 examples using the OpenAI API.

However, all 50 requests failed because the API account had exhausted its available credit/quota.

Therefore:

no LLM-as-judge scores were generated,
no unavailable scores were imputed,
and no fabricated evaluation results are reported.

This remains a limitation of the current evaluation.

A future run with an available API budget or a local/open-weight judge model would complete the LLM-based evaluation and allow direct measurement of LLM-judge versus human agreement.

17. Top 5 Failure Modes
17.1 Multi-intent Customer Messages

Some customer messages contain more than one issue.

Example:

My iPhone battery drains very quickly after the latest iOS update.

This message contains both an iOS update reference and a battery-related symptom.

The current single-label classifier may select ios_update even when the primary problem is battery drain.

Hypothesis

The classifier is currently required to select one intent for every message.

Future Improvement

Introduce:

multi-label intent classification,
primary and secondary intent detection,
symptom extraction,
and intent hierarchy.
17.2 Overprediction of performance_crash

The model frequently predicts performance_crash for broad or ambiguous technical complaints.

Hypothesis

performance_crash is the largest class in the training data and covers a relatively broad range of technical issues.

Future Improvement

Use:

hard-negative mining,
better minority-class sampling,
additional manually labelled examples,
and class-specific decision thresholds.
17.3 Vague Customer Messages

Some messages contain very little useful context.

Examples:

It doesn't work.

Having an issue with my phone.

For such messages, both classification and retrieval become unreliable.

Future Improvement

Instead of immediately generating a troubleshooting response, the agent should ask a clarification question.

Example:

Could you tell me which Apple device you're using
and what exactly is not working?
17.4 Retrieval Evidence Can Be Topically Related but Not Specific Enough

Some retrieved conversations share broad vocabulary with the customer's message but do not provide a sufficiently specific historical resolution.

For example, a battery-related query may retrieve another battery conversation but with a different underlying cause.

Hypothesis

TF-IDF similarity captures lexical overlap but does not fully understand semantic cause or resolution.

Future Improvement

Use:

sentence embeddings,
semantic vector search,
hybrid BM25 + embedding retrieval,
and reranking with a cross-encoder.
17.5 Generic Reply Drafts

Some generated replies are safe but too generic.

This is reflected in the manual evaluation, where:

Helpfulness averaged 0.63 / 2
Actionability averaged 0.40 / 2
Hypothesis

The current response templates intentionally prioritize safety and avoid unsupported troubleshooting steps.

Future Improvement

Use retrieved resolution patterns to produce more specific next steps while maintaining evidence checks.

18. What Is Misleading About My Headline Number?

The most attractive number in the system is the 88.37% auto-handle intent accuracy.

However, this number should not be interpreted as saying that the complete support agent is 88.37% accurate.

It is measured only on the 43 cases that passed the conservative auto-handle policy.

Overall:

220 cases were evaluated.
Only 43 were auto-handled.
177 were escalated.
38 of the 43 auto-handled cases had the correct predicted intent.

Therefore, the system achieves high confidence on a relatively small subset of cases by escalating uncertain requests.

The more representative overall classification accuracy is 41.82%.

This distinction is important because optimizing only for the headline auto-handle accuracy could hide poor coverage.

19. What I Would Do Next Week

The next iteration would focus on improving both classification quality and useful automation coverage.

1. Improve intent classification
Add more manually labelled training examples.
Add other_unclear as a real training class.
Use hard-negative mining.
Tune per-class thresholds.
Evaluate stronger models such as linear SVM or transformer-based classifiers.
2. Improve retrieval

Replace or complement TF-IDF with:

sentence embeddings,
vector search,
hybrid lexical + semantic retrieval,
cross-encoder reranking.
3. Improve response quality

Use retrieved resolutions to generate:

more specific troubleshooting steps,
clearer explanations,
better clarification questions,
and concise next actions.
4. Improve escalation

Create intent-specific risk policies.

For example:

billing/account issues → stricter escalation,
simple known troubleshooting → higher automation threshold,
unclear requests → clarification first.
5. Complete judge evaluation

Run the 50-example LLM-as-judge evaluation with a working API budget or local judge model and measure agreement with manual evaluation.

20. Decision Log
Decision	Reason
Selected AppleSupport	Large amount of support data and clear brand identity
Used 12-intent taxonomy	Small enough to manage while covering common support topics
Kept escalation separate from intent	A request can be understandable but still unsafe to auto-handle
Used 220-example golden set	Meets the required evaluation range and improves coverage
Excluded golden examples from training	Prevents direct evaluation leakage
Used TF-IDF for initial classifier	Fast, inexpensive and reproducible
Used Logistic Regression	Strong lightweight baseline for text classification
Added keyword baseline	Provides a simple interpretable comparison
Added majority baseline	Establishes a trivial reference point
Used historical conversations for retrieval	Grounds responses in actual brand-support behaviour
Avoided direct copying of historical replies	Reduces inappropriate response reuse
Added evidence reranking	Raw lexical similarity can retrieve partially related cases
Set confidence threshold to 0.80	Conservative automatic handling
Set similarity threshold to 0.50	Requires reasonably strong historical evidence
Escalate when evidence is weak	Safety is preferred over maximum automation
Reported auto-handle coverage separately	Prevents misleading interpretation of conditional accuracy
Did not fabricate LLM-judge results	API quota prevented valid LLM evaluation
21. Reproducibility
Requirements

Install dependencies:

pip install -r requirements.txt
Processing Pipeline

Run:

python src/preprocess.py
python src/build_threads.py
python src/discover_intents.py
python src/create_training_data.py
python src/train_intent_model.py
python src/build_resolution_index.py
Evaluation

Run:

python src/evaluate.py
python src/baseline_keyword.py
python src/evaluate_agent.py
python src/evaluate_human.py
Agent Demo

Run:

python src/support_agent.py

The evaluation artifacts are stored under:

data/processed/

The raw Kaggle dataset is downloaded separately and is not committed to the repository.

22. Conclusion

This project implements an end-to-end AI customer-support agent for AppleSupport using historical customer-support conversations.

The system combines:

intent classification,
historical retrieval,
evidence reranking,
grounded response drafting,
and conservative escalation.

The main result is that the system improves classification accuracy over both trivial and simple keyword baselines while using a conservative safety policy.

The current system intentionally sacrifices automation coverage to reduce unsafe automatic responses.

The main remaining challenges are:

-minority-intent classification,
-multi-intent messages,
-retrieval specificity,
-response actionability,
and completion of LLM-as-judge evaluation.

The next iteration should focus on stronger semantic retrieval, better intent modelling, and more useful evidence-grounded responses while preserving the current safety-first design.

