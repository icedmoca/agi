# Regular imports
import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any, Tuple, Optional
import altair as alt
import time
from core.memory import Memory
from core.audit import AuditLogger
from core.goal_router import GoalRouter
from core.evolver import Evolver
from core.chat import chat_with_llm
from core.agents.simulation import PlannerAgent, EvolverAgent, ReviewerAgent
from core.vector_memory import VectorMemory

# Session state initialization
if 'refresh' not in st.session_state:
    st.session_state.refresh = True
    st.session_state.last_refresh = time.time()
    st.session_state.chat_history = []
    if 'agents' not in st.session_state:
        st.session_state.agents = {
            'planner': PlannerAgent(),
            'evolver': EvolverAgent(),
            'reviewer': ReviewerAgent()
        }

def load_tasks() -> List[Dict[str, Any]]:
    """Load tasks from tasks.jsonl"""
    tasks = []
    tasks_file = Path("tasks.jsonl")
    if tasks_file.exists():
        with tasks_file.open() as f:
            for line in f:
                if line.strip():
                    try:
                        task = json.loads(line)
                        # Parse datetime for sorting
                        task['created_dt'] = datetime.fromisoformat(task['created'])
                        tasks.append(task)
                    except json.JSONDecodeError:
                        continue
    return tasks

def format_metadata(metadata: Dict) -> str:
    """Format metadata for display"""
    if not metadata:
        return ""
    return "\n".join(f"• {k}: {v}" for k, v in metadata.items())

def update_task_status(task_id: str, new_status: str) -> None:
    """Update task status in tasks.jsonl"""
    tasks_file = Path("tasks.jsonl")
    if not tasks_file.exists():
        return
        
    tasks = []
    with tasks_file.open() as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                if task.get("id") == task_id:
                    task["status"] = new_status
                    task["completed_at"] = datetime.now().isoformat()
                tasks.append(task)
                
    with tasks_file.open('w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

def get_memory_context(goal: str) -> str:
    """Get related memory entries for a goal"""
    memory = Memory()
    similar = memory.find_similar(goal, top_k=3)
    if not similar:
        return "No related memory entries found"
    
    return "\n".join([
        f"• {entry['goal']}: {entry['result']}" 
        for entry in similar
    ])

def chat_with_assistant(question: str) -> str:
    """Get assistant response with memory context"""
    memory = Memory()
    
    # Get relevant context from memory and recent tasks
    memory_context = memory.find_similar(question, top_k=3)
    context_str = "\n".join([
        f"Memory: {entry['goal']}: {entry['result']}"
        for entry in memory_context
    ])
    
    # Add recent tasks context
    tasks = load_tasks()
    recent_tasks = sorted(tasks, key=lambda x: x['created_dt'], reverse=True)[:5]
    tasks_str = "\n".join([
        f"Task: {task['goal']} ({task['status']})"
        for task in recent_tasks
    ])
    
    # Combine context
    full_context = f"""
    Recent Context:
    {context_str}
    
    Recent Tasks:
    {tasks_str}
    
    Question: {question}
    """
    
    try:
        response = chat_with_llm(
            system_prompt="You are a helpful AI assistant with knowledge of the system's recent actions and memory.",
            user_input=full_context
        )
        return response
    except Exception as e:
        return f"Error getting response: {str(e)}"

def rephrase_goal(goal: str) -> str:
    """Use LLM to rephrase goal more clearly"""
    prompt = f"""
    Rephrase this development goal to be clearer and more specific:
    {goal}
    
    Make it actionable and precise while preserving the original intent.
    """
    
    try:
        response = chat_with_llm(
            system_prompt="You are an expert at writing clear, specific development goals.",
            user_input=prompt
        )
        return response.strip()
    except Exception as e:
        return f"Error rephrasing goal: {str(e)}"

def evolve_task(task: Dict) -> Tuple[str, str]:
    """Execute task evolution"""
    try:
        # Route goal to files
        router = GoalRouter()
        target_files = router.route_goal_to_files(task['goal'])
        
        if not target_files:
            return "failed", "No target files identified for evolution"
            
        # Initialize evolver
        evolver = Evolver()
        
        # Evolve each file
        results = []
        for file in target_files:
            try:
                result = evolver.evolve_file(file, task['goal'])
                results.append(f"{file}: {result}")
            except Exception as e:
                results.append(f"{file}: Failed - {str(e)}")
                
        combined_result = "\n".join(results)
        return "completed", combined_result
        
    except Exception as e:
        return "failed", f"Evolution error: {str(e)}"

def load_agent_activity() -> List[Dict[str, Any]]:
    """Load agent activity log"""
    activities = []
    activity_file = Path("output/agent_activity.jsonl")
    if activity_file.exists():
        with activity_file.open() as f:
            for line in f:
                if line.strip():
                    try:
                        activity = json.loads(line)
                        activity['timestamp_dt'] = datetime.fromisoformat(activity['timestamp'])
                        activities.append(activity)
                    except json.JSONDecodeError:
                        continue
    return activities

def load_chain_executions() -> pd.DataFrame:
    """Load and parse chain execution logs"""
    log_file = Path("output/chain_execution.jsonl")
    if not log_file.exists():
        return pd.DataFrame()
        
    executions = []
    with log_file.open() as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    entry['timestamp_dt'] = datetime.fromisoformat(entry['timestamp'])
                    executions.append(entry)
                except json.JSONDecodeError:
                    continue
                    
    return pd.DataFrame(executions)

def load_task_history(task_id: str) -> List[Dict[str, Any]]:
    """Load all attempts of a task from memory"""
    memory_file = Path("memory.jsonl")
    if not memory_file.exists():
        return []
        
    history = []
    with memory_file.open() as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    if entry.get('task_id') == task_id:
                        history.append(entry)
                except json.JSONDecodeError:
                    continue
    return history

def create_retry_task(task: Dict[str, Any], chain_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a new retry task with adjusted goal"""
    # Load task history for context
    history = load_task_history(task['id'])
    
    # Increment attempt count
    current_attempt = task.get('metadata', {}).get('attempt', 1)
    
    new_task = {
        'id': f"{task['id']}_retry_{current_attempt + 1}",
        'goal': task['goal'],
        'type': task['type'],
        'status': 'pending',
        'created': datetime.now().isoformat(),
        'metadata': {
            **task.get('metadata', {}),
            'attempt': current_attempt + 1,
            'original_task_id': task['id'],
            'chain_id': chain_id,
            'tags': task.get('metadata', {}).get('tags', []) + ['retry']
        }
    }
    
    # Write to tasks file
    with Path("tasks.jsonl").open('a') as f:
        f.write(json.dumps(new_task) + '\n')
        
    return new_task

def render_execution_results():
    """Render the Execution Results tab"""
    st.header("⚡ Task Chain Execution")
    
    df = load_chain_executions()
    if df.empty:
        st.info("📝 No execution logs found yet. Chain execution results will appear here once tasks are processed.")
        return

    # Add retry buttons to the chain summary
    chain_summary = df.groupby('chain_id').agg({
        'timestamp_dt': 'min',
        'task_id': 'count',
        'status': lambda x: 'failed' if 'failed' in x.values else 'completed'
    }).reset_index()
    
    # Display each chain
    for _, chain in chain_summary.iterrows():
        chain_tasks = df[df['chain_id'] == chain['chain_id']].sort_values('timestamp_dt')
        
        with st.expander(
            f"{'❌' if chain['status'] == 'failed' else '✅'} Chain: {chain['chain_id']} ({chain['task_id']} tasks)"
        ):
            # Chain-level retry button
            if chain['status'] == 'failed':
                if st.button(f"🔄 Retry Chain {chain['chain_id']}", key=f"chain_{chain['chain_id']}"):
                    failed_tasks = chain_tasks[chain_tasks['status'] == 'failed']
                    retried_tasks = []
                    for _, task in failed_tasks.iterrows():
                        retried_task = create_retry_task(task, chain_id=chain['chain_id'])
                        retried_tasks.append(retried_task['id'])
                    st.success(f"Created retry tasks: {', '.join(retried_tasks)}")
            
            # Task details
            for _, task in chain_tasks.iterrows():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**Task:** {task['task_id']}")
                with col2:
                    st.write(f"**Status:** {'✅' if task['status'] == 'completed' else '❌'}")
                with col3:
                    # Individual task retry button
                    if task['status'] == 'failed':
                        if st.button("🔄 Retry", key=f"task_{task['task_id']}"):
                            retried_task = create_retry_task(task)
                            st.success(f"Created retry task: {retried_task['id']}")
                
                if 'goal' in task:
                    st.write(f"**Goal:** {task['goal']}")
                if 'result' in task:
                    with st.expander("Show Result"):
                        st.code(task['result'])
                st.divider()

def render_task_overview():
    """Render the Task Overview tab with statistics and charts"""
    st.header("📊 Task Overview")
    
    tasks = load_tasks()
    if not tasks:
        st.info("No tasks found. Tasks will appear here once created.")
        return
        
    # Convert to DataFrame and extract metadata fields
    df = pd.DataFrame(tasks)
    df['tags'] = df.apply(lambda x: x.get('metadata', {}).get('tags', []), axis=1)
    df['priority'] = df.apply(lambda x: x.get('metadata', {}).get('priority', 1), axis=1)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tasks", len(df))
    with col2:
        completed = len(df[df['status'] == 'completed'])
        st.metric("Completed", completed)
    with col3:
        pending = len(df[df['status'] == 'pending'])
        st.metric("Pending", pending)
    with col4:
        failed = len(df[df['status'] == 'failed'])
        st.metric("Failed", failed)
    
    # Task Status Timeline
    st.subheader("Task Status Timeline")
    timeline_data = df.copy()
    timeline_data['date'] = pd.to_datetime(timeline_data['created']).dt.date
    status_by_date = timeline_data.groupby(['date', 'status']).size().reset_index(name='count')
    
    timeline_chart = alt.Chart(status_by_date).mark_bar().encode(
        x='date:T',
        y='count:Q',
        color=alt.Color('status:N', scale=alt.Scale(
            domain=['completed', 'pending', 'failed'],
            range=['#28a745', '#ffc107', '#dc3545']
        )),
        tooltip=['date', 'status', 'count']
    ).properties(height=300)
    
    st.altair_chart(timeline_chart, use_container_width=True)
    
    # Task Distribution Charts
    cols = st.columns(2)
    
    with cols[0]:
        type_counts = df['type'].value_counts()
        fig_type = px.pie(values=type_counts.values, 
                         names=type_counts.index,
                         title="Tasks by Type")
        st.plotly_chart(fig_type, use_container_width=True)
        
    with cols[1]:
        # Flatten tags list for counting
        tag_counts = pd.Series([
            tag for tags in df['tags'] for tag in tags
        ]).value_counts()
        
        # Create DataFrame for plotly
        tag_df = pd.DataFrame({
            'Tag': tag_counts.index,
            'Count': tag_counts.values
        })
        
        fig_tags = px.bar(
            tag_df,
            x='Tag',
            y='Count',
            title="Common Tags"
        )
        st.plotly_chart(fig_tags, use_container_width=True)
    
    # Recent Tasks Table
    st.subheader("Recent Tasks")
    recent_tasks = df.sort_values('created_dt', ascending=False).head(5)
    
    for _, task in recent_tasks.iterrows():
        status_color = {
            'completed': 'green',
            'pending': 'orange',
            'failed': 'red'
        }.get(task['status'], 'grey')
        
        with st.expander(
            f"{'✅' if task['status']=='completed' else '❌' if task['status']=='failed' else '⏳'} "
            f"{task['goal'][:100]}..."
        ):
            st.write(f"**ID:** {task['id']}")
            st.write(f"**Type:** {task['type']}")
            st.write(f"**Created:** {task['created_dt'].strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Status:** :{status_color}[{task['status']}]")
            
            if task['tags']:
                st.write("**Tags:**", ", ".join(task['tags']))

def render_agent_activity():
    """Render the Agent Activity tab"""
    st.header("🤖 Agent Activity")
    
    activities = load_agent_activity()
    if not activities:
        st.info("No agent activity found yet. Activity will appear here once agents start working.")
        return
        
    # Convert to DataFrame
    activity_df = pd.DataFrame(activities)
    activity_df = activity_df.sort_values('timestamp_dt', ascending=False)
    
    # Agent filter
    agent_filter = st.selectbox(
        "Filter by Agent",
        ["All"] + sorted(activity_df['agent'].unique().tolist())
    )
    
    if agent_filter != "All":
        activity_df = activity_df[activity_df['agent'] == agent_filter]
    
    # Display activity feed
    for _, activity in activity_df.iterrows():
        with st.expander(f"{activity['agent']} - {activity['action']}"):
            st.write(f"**Task:** {activity['task_id']}")
            st.write(f"**Time:** {activity['timestamp_dt'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.code(activity['result'])

def get_similar_memories(goal: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Fetch similar memories for a given goal"""
    vector_memory = VectorMemory()
    return vector_memory.find_similar(goal, top_k=top_k)

def get_retry_history(task_id: str) -> List[Dict[str, Any]]:
    """Get all previous attempts for a task"""
    memory = Memory()
    attempts = []
    
    # Check for exact matches and prefix matches
    for entry in memory.entries:
        if (entry.get('task_id') == task_id or 
            entry.get('original_task_id') == task_id or
            (isinstance(entry.get('task_id'), str) and entry['task_id'].startswith(task_id))):
            attempts.append(entry)
            
    return sorted(attempts, key=lambda x: x.get('timestamp', ''), reverse=True)

def format_memory_entry(entry: Dict[str, Any]) -> str:
    """Format a memory entry for display"""
    timestamp = datetime.fromisoformat(entry.get('timestamp', '')).strftime('%Y-%m-%d %H:%M')
    return f"""
    🕒 {timestamp}
    📝 {entry.get('action', 'Unknown Action')}
    {'✅' if entry.get('success', False) else '❌'} Result: {entry.get('result', 'No result')}
    """

def render_task_detail(task: Dict[str, Any], key_prefix: str = "") -> None:
    """Render detailed task view with memory context"""
    with st.expander(f"{task['goal'][:100]}..."):
        # Existing task details code...
        
        # Add Memory Context
        st.subheader("🧠 Related Memory Context")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            with st.spinner("Loading memory context..."):
                similar_memories = get_similar_memories(task['goal'])
                if similar_memories:
                    for i, memory in enumerate(similar_memories, 1):
                        with st.container():
                            st.markdown(f"**Memory {i}** (Score: {memory.get('similarity_score', 0):.2f})")
                            st.markdown(format_memory_entry(memory))
                else:
                    st.info("No relevant memory context found")
        
        with col2:
            if st.button("🔍 Summarize Memory", key=f"summarize_{task['id']}"):
                with st.spinner("Generating summary..."):
                    # Format memories for LLM
                    memory_text = "\n\n".join([
                        f"Memory {i}:\n"
                        f"Goal: {mem.get('goal', 'No goal')}\n"
                        f"Result: {mem.get('result', 'No result')}\n"
                        f"Score: {mem.get('similarity_score', 0):.2f}"
                        for i, mem in enumerate(similar_memories, 1)
                    ])
                    
                    prompt = f"""
                    Analyze these related memory entries for task: "{task['goal']}"
                    
                    {memory_text}
                    
                    Provide a 2-3 sentence summary of the key insights and patterns from these memories.
                    Focus on what's most relevant to the current task.
                    """
                    
                    summary = chat_with_llm(
                        system_prompt="You are a helpful AI assistant that summarizes task-related memories.",
                        user_input=prompt
                    )
                    
                    st.markdown("### 📝 Memory Summary")
                    st.markdown(summary)
                
        # Add Retry History
        st.subheader("📑 Retry History")
        history = get_retry_history(task['id'])
        if len(history) > 1:  # More than just the current attempt
            for i, attempt in enumerate(history, 1):
                with st.expander(f"Attempt {i} - {attempt.get('timestamp', 'Unknown')}"):
                    st.markdown(format_memory_entry(attempt))
        else:
            st.info("No previous attempts found")

def render_agent_chat():
    """Render the agent chat interface in sidebar"""
    st.sidebar.subheader("💬 Agent Chat")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    # Optional task context
    tasks = load_tasks()
    task_options = ["None"] + [f"{t['id']}: {t['goal'][:50]}..." for t in tasks]
    selected_task = st.sidebar.selectbox("Ask about task:", task_options)
    
    # Chat input
    user_input = st.sidebar.text_area("Your message:")
    if st.sidebar.button("Send", key="send_chat"):
        if user_input:
            # Add task context if selected
            if selected_task != "None":
                task_id = selected_task.split(":")[0]
                task = next(t for t in tasks if t['id'] == task_id)
                context = f"Regarding task {task_id}: {task['goal']}\n\n"
                user_input = context + user_input
            
            with st.spinner("Agent is thinking..."):
                response = chat_with_llm(
                    system_prompt="You are a helpful AI assistant.",
                    user_input=user_input
                )
                
                st.session_state.chat_history.append({
                    "user": user_input,
                    "agent": response
                })
    
    # Display chat history in scrollable container
    st.sidebar.subheader("Chat History")
    chat_container = st.sidebar.container()
    with chat_container:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.text_area(
                "You:", 
                chat["user"], 
                height=70,  # Increased from 50 to meet Streamlit minimum
                disabled=True, 
                key=f"user_{i}_{hash(chat['user'])}"
            )
            st.text_area(
                "Agent:", 
                chat["agent"], 
                height=100, 
                disabled=True, 
                key=f"agent_{i}_{hash(chat['agent'])}"
            )
            st.markdown("---")

def main():
    st.set_page_config(page_title="AGI Task Dashboard", layout="wide")
    
    # Add agent chat to sidebar
    render_agent_chat()
    
    # Existing tab structure...
    tab1, tab2, tab3 = st.tabs([
        "📊 Task Overview",
        "⚡ Execution Results",
        "📝 Agent Activity"
    ])
    
    with tab1:
        render_task_overview()
    
    with tab2:
        render_execution_results()
        
    with tab3:
        render_agent_activity()

if __name__ == "__main__":
    main() 