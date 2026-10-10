"""Offline probes for the 'when things get structured or go wrong' section.

No network, no live model. Pinned against langgraph 1.2.12.
Run: python structured_cases.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy


class S(TypedDict):
    total: int


class C(TypedDict):
    total: int
    out: Annotated[list, operator.add]


print("== Two parallel nodes write the same key")
g = StateGraph(S)
g.add_node("a", lambda s: {"total": 1})
g.add_node("b", lambda s: {"total": 1})
g.add_edge(START, "a")
g.add_edge(START, "b")
g.add_edge("a", END)
g.add_edge("b", END)
try:
    g.compile().invoke({"total": 0})
except InvalidUpdateError as e:
    print("  InvalidUpdateError:", e)

print("== Command: update state and choose the next node")


def router(s):
    return Command(update={"out": ["r"]}, goto="t")


g = StateGraph(C)
g.add_node("r", router)
g.add_node("t", lambda s: {"out": ["t"]})
g.add_edge(START, "r")
g.add_edge("t", END)
print(" ", g.compile().invoke({"total": 0, "out": []}))

print("== Subgraph")
child = StateGraph(C)
child.add_node("t", lambda s: {"out": ["t"]})
child.add_edge(START, "t")
child.add_edge("t", END)

parent = StateGraph(C)
parent.add_node("child", child.compile())
parent.add_edge(START, "child")
parent.add_edge("child", END)
app = parent.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "1"}}
for event in app.stream({"total": 0, "out": []}, cfg, subgraphs=True):
    print(" ", event)

print("== Retry")
n = {"i": 0}


def flaky(s):
    n["i"] += 1
    if n["i"] < 3:
        raise ValueError("boom")
    return {"out": ["ok"]}


g = StateGraph(C)
g.add_node("f", flaky, retry_policy=RetryPolicy(retry_on=ValueError, initial_interval=0.01))
g.add_edge(START, "f")
g.add_edge("f", END)
print(" ", g.compile().invoke({"total": 0, "out": []}), f"after {n['i']} attempts")
