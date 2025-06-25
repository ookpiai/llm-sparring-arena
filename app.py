"""
LLM Sparring Arena - Compare responses from ChatGPT and Claude

DEPLOYMENT INSTRUCTIONS:
1. Set environment variables for API keys (secure)
2. Deploy to Streamlit Cloud, Railway, Render, or Heroku
3. Add API keys as secrets/environment variables in your hosting platform

SECURITY NOTE:
- API keys are now loaded from environment variables
- Safe to deploy with public repositories
- Add your keys in the deployment platform's secrets/environment variables section
"""

import streamlit as st
import openai
import anthropic
import time
import json
import os
from datetime import datetime
import pandas as pd

# API KEYS - Using environment variables for security
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Set page config
st.set_page_config(
    page_title="LLM Sparring Arena",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_responses' not in st.session_state:
    st.session_state.current_responses = {}
if 'sparring_prompt_template' not in st.session_state:
    st.session_state.sparring_prompt_template = """I queried both {model_a_name} and {model_b_name} with the following prompt:

ORIGINAL PROMPT: "{original_prompt}"

{model_a_name_upper} RESPONSE:
{model_a_response}

{model_b_name_upper} RESPONSE:
{model_b_response}

Please analyze both responses and provide:
1. Key strengths and weaknesses of each response
2. Which response better addresses the original prompt and why
3. What insights or approaches one model used that the other missed
4. Your overall assessment and recommendation
5. How you would combine the best elements of both responses

Be objective and specific in your analysis."""

def initialize_clients():
    """Initialize API clients"""
    try:
        if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
            return None, None, False
            
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return openai_client, anthropic_client, True
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return None, None, False

def call_openai(client, prompt, model="gpt-4.1"):
    """Call OpenAI API"""
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7
        )
        end_time = time.time()
        
        return {
            "content": response.choices[0].message.content,
            "model": model,
            "tokens": response.usage.total_tokens,
            "time": end_time - start_time,
            "cost": estimate_openai_cost(response.usage.total_tokens, model)
        }
    except Exception as e:
        return {"error": str(e)}

def call_anthropic(client, prompt, model="claude-sonnet-4-20250514"):
    """Call Anthropic API"""
    try:
        start_time = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        
        return {
            "content": response.content[0].text,
            "model": model,
            "tokens": response.usage.input_tokens + response.usage.output_tokens,
            "time": end_time - start_time,
            "cost": estimate_anthropic_cost(response.usage.input_tokens, response.usage.output_tokens, model)
        }
    except Exception as e:
        return {"error": str(e)}

def estimate_openai_cost(tokens, model):
    """Estimate OpenAI API cost"""
    # Updated pricing (as of 2025) - per 1k tokens
    cost_per_1k = {
        "gpt-4.1": 0.003,  # Latest and most capable (estimated)
        "gpt-4.1-mini": 0.0015,  # Faster and cheaper
        "gpt-4.1-nano": 0.0005,  # Smallest version
        "gpt-4o": 0.0025,  # Current stable
        "gpt-4o-2024-11-20": 0.0025,  # Specific version
        "gpt-4o-mini": 0.00015,  # Mini version
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.002
    }
    return (tokens / 1000) * cost_per_1k.get(model, 0.003)

def estimate_anthropic_cost(input_tokens, output_tokens, model):
    """Estimate Anthropic API cost"""
    # Updated pricing (as of 2025) - per 1k tokens
    if "claude-sonnet-4" in model:
        input_cost = (input_tokens / 1000) * 0.003
        output_cost = (output_tokens / 1000) * 0.015
    elif "claude-3-7-sonnet" in model:
        input_cost = (input_tokens / 1000) * 0.003
        output_cost = (output_tokens / 1000) * 0.015
    elif "claude-3-5-sonnet" in model:
        input_cost = (input_tokens / 1000) * 0.003
        output_cost = (output_tokens / 1000) * 0.015
    elif "claude-3-opus" in model:
        input_cost = (input_tokens / 1000) * 0.015
        output_cost = (output_tokens / 1000) * 0.075
    else:  # haiku and others
        input_cost = (input_tokens / 1000) * 0.00025
        output_cost = (output_tokens / 1000) * 0.00125
    return input_cost + output_cost

def create_comparison_prompt(original_prompt, model_a_response, model_b_response, model_a_name, model_b_name):
    """Create a prompt for model comparison using the customizable template"""
    template = st.session_state.sparring_prompt_template
    return template.format(
        original_prompt=original_prompt,
        model_a_response=model_a_response,
        model_b_response=model_b_response,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        model_a_name_upper=model_a_name.upper(),
        model_b_name_upper=model_b_name.upper()
    )

# Sidebar for settings
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model Selection
    st.subheader("Model Selection")
    openai_model = st.selectbox(
        "OpenAI Model",
        ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4-turbo"],
        index=0,
        help="GPT-4.1 is the latest and most capable model (June 2025)"
    )
    
    anthropic_model = st.selectbox(
        "Anthropic Model", 
        ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250224", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        index=0,
        help="Claude Sonnet 4 is the latest and most capable model (May 2025)"
    )
    
    # Advanced Settings
    st.subheader("Advanced Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 500, 4000, 2000, 100)
    
    # Sparring Prompt Customization
    st.subheader("⚔️ Custom Sparring Prompt")
    with st.expander("📝 Edit Sparring Prompt Template", expanded=False):
        st.markdown("**Available placeholders:**")
        st.code("""
{original_prompt} - The user's original question
{model_a_name} - First model name (e.g., "ChatGPT (gpt-4.1)")
{model_b_name} - Second model name (e.g., "Claude (claude-sonnet-4)")
{model_a_name_upper} - First model name in UPPERCASE
{model_b_name_upper} - Second model name in UPPERCASE
{model_a_response} - First model's response
{model_b_response} - Second model's response
        """)
        
        new_template = st.text_area(
            "Sparring Prompt Template:",
            value=st.session_state.sparring_prompt_template,
            height=300,
            help="Customize how models analyze each other's responses. Use the placeholders above."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Template"):
                st.session_state.sparring_prompt_template = new_template
                st.success("✅ Template saved!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Default"):
                st.session_state.sparring_prompt_template = """I queried both {model_a_name} and {model_b_name} with the following prompt:

ORIGINAL PROMPT: "{original_prompt}"

{model_a_name_upper} RESPONSE:
{model_a_response}

{model_b_name_upper} RESPONSE:
{model_b_response}

Please analyze both responses and provide:
1. Key strengths and weaknesses of each response
2. Which response better addresses the original prompt and why
3. What insights or approaches one model used that the other missed
4. Your overall assessment and recommendation
5. How you would combine the best elements of both responses

Be objective and specific in your analysis."""
                st.success("✅ Reset to default!")
                st.rerun()
    
    # API Status
    st.subheader("🔑 API Status")
    if OPENAI_API_KEY:
        st.success("✅ OpenAI API Key Configured")
    else:
        st.error("❌ OpenAI API Key Missing")
        
    if ANTHROPIC_API_KEY:
        st.success("✅ Anthropic API Key Configured")
    else:
        st.error("❌ Anthropic API Key Missing")
    
    # Clear conversation
    if st.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.session_state.current_responses = {}
        st.rerun()

# Main interface
st.title("🥊 LLM Sparring Arena")
st.markdown("Compare responses from **GPT-4.1** and **Claude Sonnet 4**, then let them critique each other!")
st.info("🚀 **Updated with the latest 2025 models!** GPT-4.1 and Claude Sonnet 4 are now the default choices.")

# Check if API keys are configured
openai_client, anthropic_client, clients_ready = initialize_clients()

if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
    st.error("❌ **API Keys Missing!** Please configure your environment variables:")
    st.code("""
# Add these environment variables in your deployment platform:
OPENAI_API_KEY=your-actual-openai-key-here
ANTHROPIC_API_KEY=your-actual-anthropic-key-here

# For local development, you can set them in your terminal:
# Windows:
set OPENAI_API_KEY=your-key-here
set ANTHROPIC_API_KEY=your-key-here

# Mac/Linux:
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
    """)
    st.info("💡 **For Streamlit Cloud:** Add these keys in your app's Settings → Secrets section")
    st.stop()

if not clients_ready:
    st.error("❌ Failed to initialize API clients. Please check your API keys.")
    st.stop()

# Main prompt input
st.subheader("💭 Enter Your Prompt")
user_prompt = st.text_area(
    "What would you like to ask both models?",
    height=100,
    placeholder="Enter your prompt here..."
)

# Action buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Query Both Models", type="primary", disabled=not user_prompt.strip()):
        if user_prompt.strip():
            with st.spinner("Querying both models..."):
                # Call both APIs simultaneously
                openai_response = call_openai(openai_client, user_prompt, openai_model)
                anthropic_response = call_anthropic(anthropic_client, user_prompt, anthropic_model)
                
                # Store responses
                st.session_state.current_responses = {
                    'prompt': user_prompt,
                    'openai': openai_response,
                    'anthropic': anthropic_response,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add to conversation history
                st.session_state.conversation_history.append(st.session_state.current_responses)

with col2:
    if st.button("⚔️ Start Sparring Round", disabled=not st.session_state.current_responses):
        if st.session_state.current_responses:
            current = st.session_state.current_responses
            
            if 'error' not in current['openai'] and 'error' not in current['anthropic']:
                with st.spinner("Models are analyzing each other's responses..."):
                    # Create comparison prompts
                    comparison_prompt = create_comparison_prompt(
                        current['prompt'],
                        current['openai']['content'],
                        current['anthropic']['content'],
                        f"ChatGPT ({openai_model})",
                        f"Claude ({anthropic_model})"
                    )
                    
                    # Get comparisons
                    openai_comparison = call_openai(openai_client, comparison_prompt, openai_model)
                    anthropic_comparison = call_anthropic(anthropic_client, comparison_prompt, anthropic_model)
                    
                    # Store comparison results
                    current['comparisons'] = {
                        'openai_comparison': openai_comparison,
                        'anthropic_comparison': anthropic_comparison
                    }
                    
                    st.session_state.current_responses = current

with col3:
    if st.button("📊 Export Results", disabled=not st.session_state.conversation_history):
        if st.session_state.conversation_history:
            # Create export data
            export_data = []
            for entry in st.session_state.conversation_history:
                export_data.append({
                    'timestamp': entry['timestamp'],
                    'prompt': entry['prompt'],
                    'openai_response': entry['openai'].get('content', 'Error'),
                    'anthropic_response': entry['anthropic'].get('content', 'Error'),
                    'openai_tokens': entry['openai'].get('tokens', 0),
                    'anthropic_tokens': entry['anthropic'].get('tokens', 0),
                    'openai_cost': entry['openai'].get('cost', 0),
                    'anthropic_cost': entry['anthropic'].get('cost', 0)
                })
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"llm_sparring_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Display current responses
if st.session_state.current_responses:
    current = st.session_state.current_responses
    
    st.divider()
    st.subheader("📋 Current Round Results")
    
    # Original responses
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🤖 ChatGPT ({openai_model})")
        if 'error' in current['openai']:
            st.error(f"Error: {current['openai']['error']}")
        else:
            st.markdown(current['openai']['content'])
            st.caption(f"⏱️ {current['openai']['time']:.2f}s | 🎯 {current['openai']['tokens']} tokens | 💰 ${current['openai']['cost']:.4f}")
    
    with col2:
        st.markdown(f"### 🧠 Claude ({anthropic_model})")
        if 'error' in current['anthropic']:
            st.error(f"Error: {current['anthropic']['error']}")
        else:
            st.markdown(current['anthropic']['content'])
            st.caption(f"⏱️ {current['anthropic']['time']:.2f}s | 🎯 {current['anthropic']['tokens']} tokens | 💰 ${current['anthropic']['cost']:.4f}")
    
    # Display comparisons if available
    if 'comparisons' in current:
        st.divider()
        st.subheader("⚔️ Sparring Round - Model Comparisons")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 ChatGPT's Analysis")
            comp = current['comparisons']['openai_comparison']
            if 'error' in comp:
                st.error(f"Error: {comp['error']}")
            else:
                st.markdown(comp['content'])
                st.caption(f"⏱️ {comp['time']:.2f}s | 🎯 {comp['tokens']} tokens | 💰 ${comp['cost']:.4f}")
        
        with col2:
            st.markdown("### 🧠 Claude's Analysis")
            comp = current['comparisons']['anthropic_comparison']
            if 'error' in comp:
                st.error(f"Error: {comp['error']}")
            else:
                st.markdown(comp['content'])
                st.caption(f"⏱️ {comp['time']:.2f}s | 🎯 {comp['tokens']} tokens | 💰 ${comp['cost']:.4f}")

# Conversation History
if st.session_state.conversation_history:
    st.divider()
    st.subheader("📚 Conversation History")
    
    # Summary statistics
    total_queries = len(st.session_state.conversation_history)
    total_openai_cost = sum(entry['openai'].get('cost', 0) for entry in st.session_state.conversation_history)
    total_anthropic_cost = sum(entry['anthropic'].get('cost', 0) for entry in st.session_state.conversation_history)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", total_queries)
    with col2:
        st.metric("OpenAI Cost", f"${total_openai_cost:.4f}")
    with col3:
        st.metric("Anthropic Cost", f"${total_anthropic_cost:.4f}")
    
    # Show history (most recent first)
    for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
        with st.expander(f"Query {total_queries - i}: {entry['prompt'][:50]}..." if len(entry['prompt']) > 50 else f"Query {total_queries - i}: {entry['prompt']}"):
            st.markdown(f"**Timestamp:** {entry['timestamp']}")
            st.markdown(f"**Prompt:** {entry['prompt']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**ChatGPT Response:**")
                if 'error' in entry['openai']:
                    st.error(entry['openai']['error'])
                else:
                    st.markdown(entry['openai']['content'][:200] + "..." if len(entry['openai']['content']) > 200 else entry['openai']['content'])
            
            with col2:
                st.markdown("**Claude Response:**")
                if 'error' in entry['anthropic']:
                    st.error(entry['anthropic']['error'])
                else:
                    st.markdown(entry['anthropic']['content'][:200] + "..." if len(entry['anthropic']['content']) > 200 else entry['anthropic']['content'])

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🥊 LLM Sparring Arena - Compare, Contrast, and Improve with AI Models</p>
    <p><small>🚀 Now featuring the latest models: GPT-4.1 & Claude Sonnet 4 (2025)</small></p>
    <p><small>🔐 Securely deployed with environment variables</small></p>
</div>
""", unsafe_allow_html=True)