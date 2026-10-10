# LangGraph internals: samples

Runnable code for my blog post **What LangGraph actually builds: Runnables, channels, bubble-up, agents and Deep Agents**.

Everything runs offline: no network, no API key, no live model (a scripted fake chat model stands in for the LLM).

| File | What it shows |
|------|---------------|
| `langgraph_primitives_demo.py` | `Runnable`, what `compile()` builds (channels and nodes), the superstep loop, an agent as a graph, bubble-up exceptions and `interrupt`, Deep Agents, reducers, checkpointer and store |
| `structured_cases.py` | Same-key write conflicts, `Command(update, goto)`, subgraphs with `stream(subgraphs=True)`, `RetryPolicy` |

## Run

```bash
pip install -r requirements.txt
python langgraph_primitives_demo.py
python structured_cases.py
```

Versions are pinned (langgraph 1.2.12, langchain-core 1.6.6, langchain 1.4.3, deepagents 0.7.20). LangGraph internals move between releases, so treat the class names the scripts print as a snapshot.
