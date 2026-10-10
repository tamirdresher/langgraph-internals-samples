"""Offline tour of what LangGraph builds from a graph, an agent and a deep agent.

No emulator, no network, no live model. Pinned against langgraph 1.2.12,
langchain 1.4.3, deepagents 0.7.20.
Run: python langgraph_primitives_demo.py
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


def show(g):
    print("  nodes   :", [n for n in g.nodes if n != "__start__"])
    print("  channels:", {k: type(v).__name__ for k, v in g.channels.items()
                          if not k.startswith("branch:")})
    for n, node in g.nodes.items():
        print(f"  node {n!r}: triggers={node.triggers} bound={type(node.bound).__name__}"
              f" writers={len(node.writers)}")
    print("  edges   :", [(e.source, e.target, "cond" if e.conditional else "")
                          for e in g.get_graph().edges])


class S(TypedDict):
    total: int
    visited: Annotated[list, operator.add]


def a(s):
    return {"total": s["total"] * 2, "visited": ["a"]}


def b(s):
    return {"total": s["total"] + 1, "visited": ["b"]}


print("== 0. Runnable: the only contract")
chain = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 10)
print("  chain type:", type(chain).__name__, "->", chain.invoke(1))

print("== 1. Two-node graph")
g = StateGraph(S)
g.add_node("a", a)
g.add_node("b", b)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)
app = g.compile()
print("  compiled type:", type(app).__name__, "| MRO:", [c.__name__ for c in type(app).__mro__[:3]])
show(app)
print("  invoke :", app.invoke({"total": 10, "visited": []}))
print("  updates:", list(app.stream({"total": 10, "visited": []}, stream_mode="updates")))

print("== 2. Bubble-up: interrupt() is an exception")


def ask(s):
    answer = interrupt("approve?")
    return {"visited": [f"answer={answer}"]}


h = StateGraph(S)
h.add_node("ask", ask)
h.add_edge(START, "ask")
h.add_edge("ask", END)
happ = h.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t"}}
print("  first call :", happ.invoke({"total": 0, "visited": []}, cfg))
print("  next       :", happ.get_state(cfg).next)
print("  resumed    :", happ.invoke(Command(resume="yes"), cfg))
print("  hierarchy  :", [c.__name__ for c in GraphInterrupt.__mro__[:3]], "| bubble-up base:",
      GraphBubbleUp.__name__)


print("== 2b. Checkpoints: one row per superstep")
saver = InMemorySaver()
capp = g.compile(checkpointer=saver)
ccfg = {"configurable": {"thread_id": "t1"}}
capp.invoke({"total": 10, "visited": []}, ccfg)
for tup in saver.list(ccfg):
    c = tup.checkpoint
    print(f"  step={tup.metadata['step']:>2} source={tup.metadata['source']:<5}"
          f" values={c['channel_values']} updated={sorted(c['updated_channels'] or [])}"
          f" pending_writes={[w[1] for w in tup.pending_writes]}")
last = saver.get_tuple(ccfg).checkpoint
print("  versions_seen keys:", sorted(last["versions_seen"]))
print("  channel_versions (shortened):", {k: v.split(".")[0].lstrip("0") for k, v in last["channel_versions"].items()})
print("  saver API:", [m for m in ("put", "put_writes", "get_tuple", "list") if hasattr(saver, m)], "serde:", type(saver.serde).__name__)
print("  StateSnapshot fields:", capp.get_state(ccfg)._fields)
before_b = [h for h in capp.get_state_history(ccfg) if h.next == ("b",)][0]
print("  time travel: state before b =", before_b.values, "-> invoke(None, that config) =", capp.invoke(None, before_b.config))
from langgraph.store.memory import InMemoryStore  # noqa: E402
store = InMemoryStore()
store.put(("users", "tamir"), "pref", {"voice": "first person"})
print("  Store (cross-thread memory):", store.get(("users", "tamir"), "pref").value)


class Scripted(BaseChatModel):
    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self.bind(tools=tools, **kw)

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        done = [m for m in messages if isinstance(m, ToolMessage)]
        if not done:
            msg = AIMessage(content="", id="ai-1", tool_calls=[
                {"name": "double", "args": {"n": 21}, "id": "call-1", "type": "tool_call"}])
        else:
            msg = AIMessage(content=f"answer {done[-1].content}", id="ai-2")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def double(n: int) -> str:
    """Double n."""
    return str(n * 2)


from langchain.agents import create_agent  # noqa: E402

print("== 3. create_agent is a StateGraph")
agent = create_agent(Scripted(), tools=[double])
print("  type:", type(agent).__name__)
show(agent)
out = agent.invoke({"messages": [HumanMessage("double 21")]})
print("  messages:", [(type(m).__name__, m.content or m.tool_calls) for m in out["messages"]])

from deepagents import create_deep_agent  # noqa: E402

print("== 4. create_deep_agent is the same graph with more middleware")
deep = create_deep_agent(model=Scripted(), tools=[double])
print("  type:", type(deep).__name__)
show(deep)
print("  tools the model sees:", sorted(deep.nodes["tools"].bound.tools_by_name))
