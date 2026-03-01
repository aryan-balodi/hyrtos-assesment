# 🚌 Singapore Transport Query Agent

A LangGraph-powered agentic workflow that answers natural-language queries about Singapore's public transport system using real-time data from the [LTA DataMall API](https://datamall.lta.gov.sg/content/datamall/en/dynamic-data.html).

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourname/transport-query-agent.git
cd transport-query-agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install langchain langgraph langchain-groq httpx langchain-core python-dotenv

# 4. Add your API keys
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY and LTA_API_KEY

# 5. Open the notebook
jupyter notebook transport_query_agent.ipynb
# Run all cells top to bottom (Kernel → Restart & Run All)
```

**API Keys required:**
- **Groq** — free at [console.groq.com](https://console.groq.com)
- **LTA DataMall** — register at [datamall.lta.gov.sg](https://datamall.lta.gov.sg/content/datamall/en/request-for-api.html)

---

## How LangGraph Powers This Agent

### Why LangGraph (and not just LangChain or raw API calls)?

| Option | Problem |
|--------|---------|
| Raw LLM + manual API calls | You hardcode every decision — "if user says bus, call bus API". Brittle, no reasoning. |
| LangChain `LLMChain` | Single-pass: one prompt → one response. Can't loop or use tools dynamically. |
| LangChain `AgentExecutor` | Works, but the execution graph is opaque and hard to inspect or extend. |
| **LangGraph `create_react_agent`** | Explicit stateful graph. Every node (agent, tools) is a first-class citizen. Full message history is inspectable at each step. Natively supports loops, conditional edges, and future extensions like memory or human-in-the-loop. |

### How LangGraph Works Here — Step by Step

LangGraph models agent execution as a **directed graph of nodes connected by edges**. `create_react_agent` compiles this graph for you:

```
┌─────────────────────────────────────────────┐
│              LangGraph StateGraph            │
│                                              │
│   START                                      │
│     │                                        │
│     ▼                                        │
│  ┌──────────┐   has tool_calls?   ┌────────┐ │
│  │  agent   │ ──────────────────► │ tools  │ │
│  │  node    │ ◄────────────────── │  node  │ │
│  └──────────┘   returns results   └────────┘ │
│     │                                        │
│     │ no tool_calls (final answer)           │
│     ▼                                        │
│   END                                        │
└─────────────────────────────────────────────┘
```

**State** is a list of messages (`HumanMessage`, `AIMessage`, `ToolMessage`) that grows with each step. The LLM always sees the full conversation history.

**Agent node** — the LLM receives the current message history + the system prompt + the schema of all 10 tools. It returns either:
- An `AIMessage` with `tool_calls` → graph routes to the tools node
- An `AIMessage` with plain text content → graph routes to END

**Tools node** — executes every requested tool call in parallel, appends each result as a `ToolMessage`, and routes back to the agent node.

**The loop runs until** the LLM produces a response with no tool calls.

### Why ReAct Over a Simple Intent Router?

A router maps keywords to fixed actions — "bus" → call BusArrival, "train" → call TrainAlerts. It breaks the moment the query is ambiguous or multi-intent.

ReAct lets the LLM:
1. **Reason first** — "The user asked about getting to CBD in heavy rain. I should check EWL crowd levels AND train disruptions AND PIE traffic in case they drive instead."
2. **Act** — calls all three tools
3. **Observe** — reads the results
4. **Reason again** — "No disruptions on EWL, PIE is congested, crowd is moderate. My answer: take the MRT."

This reasoning chain happens automatically. Simulation 10 demonstrates it — the agent calls three different tools in sequence without any explicit routing code.

### Why GPT-OSS 120B via Groq?

- Smaller models (Llama 8B) have an **8K tokens-per-minute** rate limit on Groq's free tier. A single multi-tool query with large API responses can easily exceed this, causing 413 errors.
- GPT-OSS 120B has a **250K TPM** limit — sufficient for all 10 simulations running sequentially.
- Smaller models also produced **malformed tool call JSON** (e.g., `{"}"`) on zero-argument tools — a known issue that required a workaround (every tool now has at least one optional parameter).

### Why Tool Docstrings Matter

The LLM never sees the tool's source code — only its **name, parameters, and docstring**. The docstring is what the model uses to decide *whether* to call a tool and *what arguments* to pass. Vague docstrings lead to wrong tool selection or incorrect arguments. Every tool here has explicit argument descriptions with examples:

```python
@tool
def get_bus_stops_info(search: str = "") -> str:
    """Search for bus stops by road name, description, or landmark.
    Args:
        search: Keyword to filter. Examples: 'Orchard', 'Lucky Plaza', 'Tampines'
    """
```

Without the examples, the model might pass `"Lucky Plaza MRT station, Orchard Road, Singapore"` instead of just `"lucky plaza"`.

---

## Evaluation Criteria

### 1. Agentic Workflow Design

The agent uses **LangGraph's ReAct (Reason + Act) pattern** — the LLM autonomously decides which tools to call, calls them, observes the results, and decides whether to fetch more data or synthesise a final answer. There is no hardcoded routing logic.

```
User Query
   ↓
Agent Node (GPT-OSS 120B via Groq)
   ↓ reasons about which tool(s) to call
Tool Execution (LTA DataMall API calls)
   ↓ observes results
Agent Node
   ↓ synthesises human-friendly response
Final Answer
```

**10 tools registered**, each wrapping a distinct LTA DataMall endpoint:

| Tool | Endpoint | Data |
|------|----------|------|
| `get_bus_arrival` | `v3/BusArrival` | Real-time ETAs at any bus stop |
| `get_train_service_alerts` | `TrainServiceAlerts` | Live MRT/LRT disruptions |
| `get_station_crowd_density` | `PCDRealTime` | Station crowd levels (low/moderate/high) |
| `get_traffic_incidents` | `TrafficIncidents` | Accidents, road works, diversions |
| `get_carpark_availability` | `CarParkAvailabilityv2` | Live parking lot counts |
| `get_taxi_availability` | `Taxi-Availability` | GPS positions of available taxis |
| `get_bus_stops_info` | `BusStops` | Paginated stop search by keyword/landmark |
| `get_passenger_volume_by_train_stations` | `PV/Train` | Monthly station ridership data |
| `get_traffic_speed_bands` | `TrafficSpeedBandsv2` | Road congestion levels by band |
| `get_bus_routes` | `BusRoutes` | Full route, stops, and first/last bus times |

The agent chains multiple tools in a single turn when needed — e.g., Simulation 10 calls crowd density + train alerts + traffic speed bands without any explicit orchestration code.

---

### 2. Variety of Inputs as Constraints

Every agent invocation receives a **dynamically generated system prompt** that injects three contextual layers before the LLM sees the user query:

| Context | How it works |
|---------|-------------|
| **Time of day** | Detects rush hour (7–9 AM / 5–7 PM), off-peak, late night, weekend — the agent proactively warns about crowding or reduced frequency |
| **Weather** | Four conditions: `clear`, `light_rain`, `heavy_rain`, `haze` — in heavy rain the agent recommends sheltered MRT routes over buses |
| **Public holidays** | Hardcoded Singapore 2025–2026 calendar — on CNY Day 1 the agent notes Sunday/holiday schedules and reduced frequency |

**10 simulations stress-test this grid:**

| # | Persona | Time | Weather | Holiday | Tools Used |
|---|---------|------|---------|---------|------------|
| 1 | Morning Commuter | 8:00 AM weekday | Clear | — | Bus Arrival |
| 2 | Rainy Day Traveller | 8:30 AM weekday | Heavy rain | — | Crowd Density |
| 3 | Weekend Explorer | 10:00 AM Sunday | Clear | — | Bus Routes |
| 4 | Driver on PIE | 9:00 AM weekday | Clear | — | Traffic Incidents + Speed Bands |
| 5 | Accessibility User | 2:00 PM weekday | Clear | — | Train Alerts + Crowd Density |
| 6 | Late Night User | 11:30 PM Friday | Clear | — | Taxi Availability |
| 7 | Holiday Planner | 10:00 AM | Clear | CNY Day 1 | Train Alerts |
| 8 | Parking Seeker | 3:00 PM Saturday | Light rain | — | Carpark Availability |
| 9 | Tourist | 11:00 AM weekday | Clear | — | Bus Stops + Bus Arrival |
| 10 | Power User | 8:15 AM weekday | Heavy rain | — | Crowd Density + Train Alerts + Speed Bands |

---

### 3. Solution Structure

The notebook is organised into clean, independent layers — each section can be read and modified without touching the others:

```
Section 2  │ Setup — dependency install, .env key loading
Section 3  │ API Layer — lta_api_request(), _safe_json(), _truncate()
           │ Tool Layer — 10 @tool-decorated functions with rich docstrings
Section 4  │ Context Layer — time/weather/holiday → dynamic system prompt
Section 5  │ Agent Layer — create_react_agent(), create_transport_agent()
           │ Runner Layer — run_query() with full execution trace output
Section 6  │ 10 simulation cells (one per user persona)
Section 7  │ Deployment considerations
```

**Key engineering decisions:**

- **`_safe_json()` + `_truncate()`** — LTA APIs return thousands of records. Feeding all of them to the LLM exceeds token limits. Responses are capped at 50 items and 3,000 characters (~750 tokens) — enough to cover the longest MRT line (EWL, 35 stations) without wasting context.
- **Optional parameter on every tool** — Tools with zero parameters caused the model to generate malformed JSON. Every tool has at least one optional `str = ""` argument so the model always produces valid JSON like `{}` or `{"area": "PIE"}`.
- **Paginated bus stop search** — `get_bus_stops_info` iterates through batches of 500 stops across up to 20 pages until it finds keyword matches, rather than returning only the first batch (which would only contain stops starting with code `01xxx`).

---

### 4. Explanation and Documentation

- **Notebook opening** — problem statement, architecture diagram, design decisions, and assumptions
- **Per-section markdown** — each code section is preceded by a markdown cell explaining its role and design rationale
- **Tool docstrings** — every tool has a detailed docstring that the LLM reads to decide which tool to use; docstrings include argument descriptions and valid input examples
- **Simulation matrix** — a table mapping each simulation to its persona, time, weather, holiday, and expected tools
- **Verbose execution trace** — `run_query()` prints every tool call with arguments, the API response snippet, tools-used summary, and the final answer for each simulation

---

### 5. Deployment Considerations

See **Section 7** of the notebook for the full production architecture. Summary:

**Containerisation**
```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Caching strategy (Redis with per-endpoint TTLs)**

| Data | TTL | Reason |
|------|-----|--------|
| Bus Arrival | 30–60 s | Changes every bus cycle |
| Bus Stops / Routes | 24 h | Static reference data |
| Train Alerts | 60 s | Near real-time disruptions |
| Crowd Density | 5 min | Updates on 30-min intervals |
| Carpark Availability | 2 min | Moderate change rate |
| Traffic Incidents | 1 min | Safety-critical |

**Scaling** — Stateless LangGraph workers behind a FastAPI app, horizontal scaling with Kubernetes/Celery.

**Security** — API keys in AWS Secrets Manager; input sanitization before LLM; rate limiting (60 req/min per user); prompt injection defense via tool argument schema validation.

**Observability** — LangSmith for per-run agent traces; Prometheus + Grafana for system metrics; structured JSON logging with correlation IDs.

**Future enhancements**
- Real NEA weather API instead of simulated conditions
- Multi-turn conversation memory via LangGraph checkpointing
- Geospatial bus stop lookup (PostGIS) to replace paginated keyword search
- Multilingual support — Mandarin, Malay, Tamil

---

## Assumptions

- Weather data is **simulated** — real deployment would call the [NEA Realtime Weather API](https://data.gov.sg/dataset/realtime-weather-readings)
- Singapore public holiday calendar is **hardcoded** for 2025–2026
- LTA DataMall `FacilitiesMaintenance` (lift/escalator status) is **not available** in the open-tier API and is therefore not included
- The `BusRoutes` endpoint returns all routes island-wide; the tool filters client-side by `service_no` since the API does not support server-side filtering

---

## Project Structure

```
transport-query-agent/
├── transport_query_agent.ipynb   # Main notebook — source + all simulation outputs
├── .env.example                  # API key template (copy to .env and fill in)
├── .gitignore                    # Excludes .env, .venv, __pycache__, checkpoints
└── README.md                     # This file
```
