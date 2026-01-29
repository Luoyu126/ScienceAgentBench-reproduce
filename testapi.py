import anthropic
import os

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    base_url=os.environ.get("ANTHROPIC_API_BASE_URL")
)
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": """You are an expert Python programming assistant that helps scientist users to write high-quality code to solve their tasks.\nGiven a user request, you are expected to write a complete program that accomplishes the requested task and save any outputs in the correct format.\nPlease wrap your program in a code block that specifies the script type, python. For example:\n```python\nprint("Hello World!")\n```\n\nPlease keep your response concise and do not use a code block if it\'s not intended to be executed.\nPlease do not suggest a few line changes, incomplete program outline, or partial code that requires the user to modify.\nPlease do not use any interactive Python commands in your program, such as `!pip install numpy`, which will cause execution errors.\n\nHere\'s the user request you need to work on:\nTrain a multitask model on the Clintox dataset to predict a drug\'s toxicity and FDA approval status. Save the test set predictions, including the SMILES representation of drugs and the probability of positive labels, to "pred_results/clintox_test_pred.csv".\nYou can access the dataset at `benchmark/datasets/clintox/`. Here is the directory structure of the dataset:\n```\n|-- clintox/\n|---- clintox_test.csv\n|---- clintox_train.csv\n```\nHere are some helpful previews for the dataset file(s):\n[START Preview of clintox/clintox_train.csv]\nsmiles,FDA_APPROVED,CT_TOX\nCCC(/C=C/Cl)(C#C)O,1,0\nC[C@H]1C[C@H]2[C@@H]3CC[C@@H]([C@]3(C[C@@H]([C@@H]2[C@@]4(C1=CC(=O)CC4)C)O)C)C(=O)C,1,0\nC[C@@H]1CCN([C@H](C1)C(=O)[O-])C(=O)[C@H](CCC[NH+]=C(N)N)NS(=O)(=O)c2cccc3c2NC[C@@H](C3)C,1,0\n...\n[END Preview of clintox/clintox_train.csv]'"""}
    ]
)
print(message.content)