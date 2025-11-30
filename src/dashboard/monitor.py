"""Teacher monitoring dashboard using Streamlit."""

import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import sys
import os

# Add the project root folder to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

# Now you can safely import main.py
from main import start_honeypot_components, stop_honeypot_components



# Configure page
st.set_page_config(
    page_title="AI Honeypot Monitor",
    page_icon="🍯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_dashboard():
    """Initialize dashboard components."""
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    # Initialize honeypot status
    if 'honeypot_running' not in st.session_state:
        st.session_state.honeypot_running = False

def get_activity_data() -> Dict[str, Any]:
    """Get current activity data from logger."""
    try:
        from src.utils.logger import get_logger
        logger = get_logger()
        
        activities = logger.get_recent_activities(100)
        stats = logger.get_activity_stats()
        
        return {
            "activities": activities,
            "stats": stats,
            "last_updated": datetime.now()
        }
    except Exception as e:
        st.error(f"Error fetching activity data: {e}")
        return {"activities": [], "stats": {}, "last_updated": datetime.now()}

def display_overview_metrics(stats: Dict[str, Any]):
    """Display overview metrics."""
    st.subheader("📊 Activity Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Activities", 
            value=stats.get("total", 0),
            delta=f"+{stats.get('total', 0) - st.session_state.get('prev_total', 0)}" if 'prev_total' in st.session_state else None
        )
    
    with col2:
        st.metric(
            label="Unique IPs", 
            value=stats.get("unique_ips", 0),
            delta=f"+{stats.get('unique_ips', 0) - st.session_state.get('prev_ips', 0)}" if 'prev_ips' in st.session_state else None
        )
    
    with col3:
        connections = stats.get("by_type", {}).get("connection", 0)
        st.metric("SSH Connections", value=connections)
    
    with col4:
        ai_generations = stats.get("by_type", {}).get("ai_generation", 0)
        st.metric("AI Generations", value=ai_generations)
    
    # Store previous values
    st.session_state.prev_total = stats.get("total", 0)
    st.session_state.prev_ips = stats.get("unique_ips", 0)

def display_activity_timeline(activities: List[Dict[str, Any]]):
    """Display activity timeline chart."""
    st.subheader("⏰ Activity Timeline")
    
    if not activities:
        st.info("No activities recorded yet. Start the honeypot and wait for connections!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(activities)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.floor('h') # Fixed FutureWarning
    
    # Activity by hour
    hourly_activity = df.groupby(['hour', 'type']).size().reset_index(name='count')
    
    if not hourly_activity.empty:
        fig = px.bar(
            hourly_activity, 
            x='hour', 
            y='count', 
            color='type',
            title="Activity by Hour and Type",
            height=400
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Number of Activities")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for timeline chart yet.")

def display_ai_generation_stats(activities: List[Dict[str, Any]]):
    """Display AI generation statistics."""
    st.subheader("🤖 AI Generation Activity")
    
    # Filter AI generation activities
    ai_activities = [a for a in activities if a.get('type') == 'ai_generation']
    
    if not ai_activities:
        st.info("No AI generations recorded yet.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generator type distribution
        generators = {}
        for activity in ai_activities:
            gen_type = activity.get('generator', 'unknown')
            generators[gen_type] = generators.get(gen_type, 0) + 1
        
        if generators:
            fig = px.pie(
                values=list(generators.values()), 
                names=list(generators.keys()),
                title="AI Generators Used"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Generation times
        generation_times = [a.get('generation_time', 0) for a in ai_activities if 'generation_time' in a]
        
        if generation_times:
            fig = px.histogram(
                x=generation_times,
                title="AI Generation Time Distribution",
                labels={'x': 'Generation Time (seconds)', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

def display_recent_activity_feed(activities: List[Dict[str, Any]]):
    """Display recent activity feed."""
    st.subheader("📋 Live Activity Feed")
    
    if not activities:
        st.info("No activities to display.")
        return
    
    # Show most recent activities
    recent_activities = sorted(activities, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]
    
    for activity in recent_activities:
        timestamp = activity.get('timestamp', 'Unknown')
        activity_type = activity.get('type', 'unknown')
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime('%H:%M:%S')
        except:
            time_str = timestamp[:8] if len(timestamp) > 8 else timestamp
        
        # Create activity message based on type
        if activity_type == 'connection':
            client_ip = activity.get('client_ip', 'Unknown')
            connection_type = activity.get('connection_type', 'SSH')
            message = f"🔗 **{connection_type} Connection** from `{client_ip}`"
            
        elif activity_type == 'command':
            client_ip = activity.get('client_ip', 'Unknown')
            command = activity.get('command', 'unknown')
            response_type = activity.get('response_type', 'static')
            message = f"💻 **Command executed:** `{command}` by `{client_ip}` ({response_type})"
            
        elif activity_type == 'ai_generation':
            generator = activity.get('generator', 'AI')
            content_type = activity.get('content_type', 'content')
            gen_time = activity.get('generation_time', 0)
            message = f"🤖 **{generator}** generated `{content_type}` in {gen_time:.2f}s"
            
        elif activity_type == 'file_access':
            client_ip = activity.get('client_ip', 'Unknown')
            file_path = activity.get('file_path', 'unknown')
            access_type = activity.get('access_type', 'read')
            message = f"📁 **File {access_type}:** `{file_path}` by `{client_ip}`"
            
        else:
            message = f"ℹ️ **{activity_type}:** {str(activity)[:100]}..."
        
        # Display activity with timestamp
        with st.container():
            st.markdown(f"**{time_str}** - {message}")
            
        st.markdown("---")

def display_ai_model_status():
    """Display AI model training and status information."""
    st.subheader("🧠 AI Model Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**GPT-2 Text Generator**")
        try:
            from src.ai_engine.text_generator import get_text_generator
            text_gen = get_text_generator()
            st.success("✅ Loaded and Ready")
            st.info(f"Model: {text_gen.tokenizer.name_or_path}")
        except Exception as e:
            st.error("❌ Not Available")
            st.error(f"Error: {str(e)[:50]}...")
    
    with col2:
        st.markdown("**CTGAN Tabular Generator**")
        try:
            from src.ai_engine.tabular_generator import get_tabular_generator
            tabular_gen = get_tabular_generator()
            st.success("✅ Loaded and Ready")
            
            # Show available tables
            tables = tabular_gen.list_available_tables()
            st.info(f"Tables: {', '.join(tables[:3])}...")
            
            # Show training status
            trained_models = len(tabular_gen.models)
            st.metric("Trained Models", trained_models)
            
        except Exception as e:
            st.error("❌ Not Available")
            st.error(f"Error: {str(e)[:50]}...")
    
    with col3:
        st.markdown("**TimeGAN Log Generator**")
        try:
            from src.ai_engine.log_generator import get_log_generator
            log_gen = get_log_generator()
            st.success("✅ Loaded and Ready")
            
            # Show model status
            trained_log_models = len(log_gen.models)
            st.metric("Trained Models", trained_log_models)
            
        except Exception as e:
            st.error("❌ Not Available")
            st.error(f"Error: {str(e)[:50]}...")

def display_synthetic_data_viewer():
    """Display synthetic data generation interface."""
    st.subheader("🗂️ Synthetic Data Viewer")
    
    try:
        from src.ai_engine.tabular_generator import get_tabular_generator
        tabular_gen = get_tabular_generator()
        
        # Table selection
        tables = tabular_gen.list_available_tables()
        selected_table = st.selectbox("Select Table to Generate:", tables)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_rows = st.slider("Number of rows to generate:", 5, 100, 20)
        
        with col2:
            if st.button("Generate Synthetic Data"):
                with st.spinner(f"Generating {num_rows} rows for {selected_table}..."):
                    synthetic_data = tabular_gen.generate_synthetic_data(selected_table, num_rows)
                    
                    if synthetic_data is not None:
                        st.success(f"Generated {len(synthetic_data)} rows successfully!")
                        st.dataframe(synthetic_data.head(10))
                        
                        # Download option
                        csv = synthetic_data.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{selected_table}_synthetic.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Failed to generate synthetic data")
        
        # Show table information
        if selected_table:
            table_info = tabular_gen.get_table_info(selected_table)
            
            with st.expander(f"📋 {selected_table.title()} Table Information"):
                st.json(table_info)
                
    except Exception as e:
        st.error(f"Error accessing synthetic data generator: {e}")

def display_log_generator_interface():
    """Display log generation interface."""
    st.subheader("📊 Log Generator Interface")
    
    try:
        from src.ai_engine.log_generator import get_log_generator
        log_gen = get_log_generator()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Generate Auth Log"):
                with st.spinner("Generating authentication logs..."):
                    auth_log = log_gen.generate_auth_log(15)
                    st.text_area("Authentication Log:", auth_log, height=200)
        
        with col2:
            if st.button("Generate System Log"):
                with st.spinner("Generating system logs..."):
                    sys_log = log_gen.generate_syslog(20)
                    st.text_area("System Log:", sys_log, height=200)
        
        with col3:
            hours = st.slider("Hours of log sequence:", 1, 24, 6)
            if st.button("Generate Log Sequence"):
                with st.spinner(f"Generating {hours} hours of log sequence..."):
                    log_sequence = log_gen.generate_log_sequence(num_hours=hours)
                    
                    # Convert to readable format
                    log_text = "\n".join([
                        f"{entry['timestamp']} [{entry['level']}] {entry['service']}: {entry['message']}"
                        for entry in log_sequence
                    ])
                    
                    st.text_area("Log Sequence:", log_text, height=300)
        
    except Exception as e:
        st.error(f"Error accessing log generator: {e}")

def main():
    """Main dashboard function."""
    st.title("🍯 AI-Driven Honeypot Simulator Dashboard")
    st.markdown("**Real-time monitoring of honeypot activities and AI generation**")
    
    # Initialize dashboard
    initialize_dashboard()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Honeypot start/stop buttons
        if st.session_state.honeypot_running:
            if st.button("⏹️ Stop Honeypot", help="Stop the SSH server and other honeypot components"):
                stop_honeypot_components()
                st.session_state.honeypot_running = False
                st.success("Honeypot components stopped.")
                time.sleep(1) # Give it a moment to update
                st.rerun()
        else:
            if st.button("▶️ Start Honeypot", help="Start the SSH server and other honeypot components"):
                start_honeypot_components()
                st.session_state.honeypot_running = True
                st.success("Honeypot components started.")
                time.sleep(1) # Give it a moment to update
                st.rerun()

        # Display current status
        status_emoji = "🟢" if st.session_state.honeypot_running else "🔴"
        st.markdown(f"**Honeypot Status:** {status_emoji} {'Running' if st.session_state.honeypot_running else 'Stopped'}")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        # Refresh interval
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        # Clear logs button
        if st.button("🗑️ Clear Logs"):
            try:
                from src.utils.logger import get_logger
                logger = get_logger()
                logger._activities.clear()
                st.success("Logs cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing logs: {e}")
        
        # Status information
        st.markdown("---")
        st.markdown("**System Status**")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Get current data
    data = get_activity_data()
    
    # Main dashboard content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 AI Models", "🗂️ Data Generator", "📊 Log Generator"])
    
    with tab1:
        # Overview metrics
        display_overview_metrics(data["stats"])
        
        # Activity timeline
        display_activity_timeline(data["activities"])
        
        # AI generation stats
        display_ai_generation_stats(data["activities"])
        
        # Recent activity feed
        display_recent_activity_feed(data["activities"])
    
    with tab2:
        # AI model status
        display_ai_model_status()
        
        # Model training interface
        st.subheader("🔧 Model Training Interface")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Train CTGAN Model**")
            table_select = st.selectbox("Select table for CTGAN training:", 
                                      ["customers", "transactions", "users", "logs"])
            epochs = st.slider("Training epochs:", 10, 100, 50)
            
            if st.button("Train CTGAN Model"):
                try:
                    from src.ai_engine.tabular_generator import get_tabular_generator
                    tabular_gen = get_tabular_generator()
                    
                    with st.spinner(f"Training CTGAN on {table_select} for {epochs} epochs..."):
                        success = tabular_gen.train_model(table_select, epochs)
                        
                    if success:
                        st.success(f"CTGAN model for {table_select} trained successfully!")
                    else:
                        st.error("Training failed!")
                        
                except Exception as e:
                    st.error(f"Training error: {e}")
        
        with col2:
            st.markdown("**Train TimeGAN Model**")
            log_model_name = st.text_input("Log model name:", value="system_logs")
            sequences = st.slider("Training sequences:", 50, 500, 200)
            log_epochs = st.slider("Log training epochs:", 10, 100, 50)
            
            if st.button("Train TimeGAN Model"):
                try:
                    from src.ai_engine.log_generator import get_log_generator
                    log_gen = get_log_generator()
                    
                    with st.spinner(f"Training TimeGAN model {log_model_name}..."):
                        success = log_gen.train_timegan_model(log_model_name, sequences, log_epochs)
                        
                    if success:
                        st.success(f"TimeGAN model {log_model_name} trained successfully!")
                    else:
                        st.error("Training failed!")
                        
                except Exception as e:
                    st.error(f"Training error: {e}")
    
    with tab3:
        display_synthetic_data_viewer()
    
    with tab4:
        display_log_generator_interface()
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
