# agents/summarizer_agent.py
"""
Build a short operator briefing from the fully-merged report dict.

The summarizer is called *after* the coordinator has produced a complete
report, so every section (radar / tweet / resource) is guaranteed to be
present.
"""
from langchain.prompts import ChatPromptTemplate
from ._llm import llm

# define our system + human messages template
_SUM_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Summarizer / Dashboard agent. "
     "Create a very detailed multi-bullet briefing for an emergency operator "
     "from the provided situation report. Give concrete numbers where possible."),
    ("human", "{context}")
])

def summarize(report: dict) -> str:
    """Return a short, human-readable briefing."""
    # 1) pull out each piece
    radar    = report["radar"]
    tweet    = report["tweet"]
    resource = report["resource"]

    # 2) build the flat text context
    context = (
        f"Tornado probability: {radar['tornado_prob']:.1%}\n"
        f"Tweet signal        : {tweet['multi']}\n"
        f"Coordinator actions : {resource}"
    )

    # 3) format into a ChatPromptValue
    prompt_value = _SUM_PROMPT.format_prompt(context=context)

    # 4) extract the actual list of messages (SystemMessage + HumanMessage)
    messages = prompt_value.to_messages()

    # 5) call the LLM directly with those messages
    response = llm(messages)

    # 6) unpack its .content (or fall back to str)
    return getattr(response, "content", str(response))
