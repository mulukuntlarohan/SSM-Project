"""
SSM Framework - Streamlit Chat Interface
=========================================

Minimal, clean chat interface with drift detection.
- Simple chat interface
- Automatic drift detection with pop-up alerts
- State extraction visualization on drift
- Agent-driven conflict resolution
"""

import os
import sys
import json
from typing import Optional, Dict, Any, List

import streamlit as st

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import GlobalStateMap, ConflictRecord
from src.agents import SSMAgent, create_agent


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SSM Chat",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Minimal styling
st.markdown("""
<style>
    .drift-alert {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .state-popup {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "drift_choice" not in st.session_state:
        st.session_state.drift_choice = None
    
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    
    if "pending_result" not in st.session_state:
        st.session_state.pending_result = None


def get_agent() -> SSMAgent:
    """Get or create the SSM agent instance."""
    if st.session_state.agent is None:
        print(f"\n[GET AGENT] Creating new agent instance")
        with st.spinner("Initializing SSM Agent..."):
            st.session_state.agent = create_agent()
        print(f"[GET AGENT] Agent created and cached in session")
    else:
        print(f"[GET AGENT] Reusing cached agent")
    return st.session_state.agent


# =============================================================================
# MINIMAL SIDEBAR
# =============================================================================

def render_sidebar():
    """Render minimal sidebar with only controls."""
    with st.sidebar:
        st.markdown("### Controls")
        
        if st.button("🔄 Reset Conversation", use_container_width=True, key="reset_btn"):
            try:
                print(f"\n[RESET BUTTON] Reset clicked")
                # Reset agent (which automatically backs up state via StateMapManager.reset())
                if "agent" in st.session_state:
                    agent = st.session_state.agent
                    if agent:
                        print(f"[RESET BUTTON] Calling agent.reset_state()")
                        agent.reset_state()
                        print(f"[RESET BUTTON] Agent state reset successfully")
                
                # Clear message history for new conversation
                print(f"[RESET BUTTON] Clearing messages and agent from session")
                st.session_state.messages = []
                st.session_state.agent = None  # Force reinit on next input
                st.session_state.drift_choice = None
                st.session_state.pending_prompt = None
                st.session_state.pending_result = None
                
                st.toast("✓ Conversation reset and backed up!", icon="✅")
                print(f"[RESET BUTTON] Session state cleared, calling st.rerun()")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")


# =============================================================================
# DRIFT DETECTION & STATE DISPLAY (POP-UP)
# =============================================================================

def show_drift_popup(extraction, conflicts: List[ConflictRecord], state: GlobalStateMap):
    """Show drift detection pop-up with extracted state and user choice."""
    st.markdown("""
    <div class="drift-alert">
        <strong>⚠️ Drift Detected</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Show extracted features
    st.markdown("**📊 Extracted Technical Features:**")
    if extraction:
        # Handle both dict and ExtractionResult objects
        if hasattr(extraction, 'extracted_state'):
            # It's an ExtractionResult object
            extracted_tech = extraction.extracted_state.get("tech", {}) if isinstance(extraction.extracted_state, dict) else {}
            if extracted_tech:
                st.markdown("**Technical Stack:**")
                for key, value in extracted_tech.items():
                    if value and value != "none":
                        st.markdown(f"- **{key}**: `{value}`")
            
            if extraction.raw_intent:
                st.markdown(f"**Intent**: {extraction.raw_intent}")
        else:
            # It's a dict
            extracted_features = extraction.get("extracted_features", {})
            if extracted_features:
                for key, value in extracted_features.items():
                    if value and value != "none":
                        st.markdown(f"- **{key}**: `{value}`")
            
            if extraction.get("raw_intent"):
                st.markdown(f"**Intent**: {extraction['raw_intent']}")
    
    # Show conflicts
    if conflicts:
        st.markdown("**⚡ Conflicts with Current State:**")
        for conflict in conflicts:
            st.markdown(f"""
            - **{conflict.field_path}**  
            Current: `{conflict.existing_value}` → New: `{conflict.proposed_value}`
            """)
    
    # User choice
    st.divider()
    print(f"\n[POPUP RENDER] About to show drift choice buttons...")
    col1, col2 = st.columns(2)
    
    with col1:
        print(f"[POPUP RENDER] Accept button rendering...")
        if st.button("✅ Accept New Values", use_container_width=True, key="accept_drift"):
            print(f"\n[BUTTON] ACCEPT clicked")
            st.session_state["drift_choice"] = "accept"
            print(f"[BUTTON] Set drift_choice = accept")
            st.toast("Accepting new values...", icon="✅")
            print(f"[BUTTON] About to call st.rerun()...")
            st.rerun()
    
    with col2:
        print(f"[POPUP RENDER] Reject button rendering...")
        if st.button("❌ Stick with Original", use_container_width=True, key="reject_drift"):
            print(f"\n[BUTTON] REJECT clicked")
            st.session_state["drift_choice"] = "reject"
            print(f"[BUTTON] Set drift_choice = reject")
            st.toast("Keeping original values...", icon="⛔")
            print(f"[BUTTON] About to call st.rerun()...")
            st.rerun()
    
    # Show current state
    with st.expander("📊 Current State Details"):
        state_dict = state.model_dump(exclude_none=True)
        # Simplify display - show only tech section
        if "tech" in state_dict:
            st.json({"tech": state_dict["tech"]})


# =============================================================================
# CHAT INTERFACE
# =============================================================================

def render_chat_interface(agent: SSMAgent):
    """Render the main chat interface."""
    st.title("🧠 SSM Chat")
    
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle pending drift choice from previous rerun before accepting new input.
    pending_prompt = st.session_state.get("pending_prompt")
    drift_choice = st.session_state.get("drift_choice")
    pending_result = st.session_state.get("pending_result") or {}

    print(f"\n[DRIFT HANDLER CHECK] State check:")
    print(f"  - pending_prompt exists: {pending_prompt is not None}")
    print(f"  - drift_choice: {drift_choice}")

    if pending_prompt:
        if drift_choice:
            print(f"\n[DRIFT HANDLER] TRIGGERED - Processing drift choice: {drift_choice}")
            print(f"[DRIFT HANDLER] Pending prompt: {pending_prompt[:50]}...")
            
            state_before = agent.get_state()
            print(f"[DRIFT HANDLER] State BEFORE processing: language={state_before.tech.language}")

            with st.chat_message("assistant"):
                try:
                    if drift_choice == "accept":
                        # User accepted: reprocess with force_merge=True to apply changes
                        with st.spinner("Accepting changes..."):
                            st.info("🔄 Accepting new values and updating state...")
                            print(f"[DRIFT HANDLER] Calling agent.process(pending_prompt, force_merge=True)...")
                            result = agent.process(pending_prompt, force_merge=True)
                        
                        state_after_merge = agent.get_state()
                        print(f"[DRIFT HANDLER] State AFTER merge: language={state_after_merge.tech.language}")

                        print(f"[DRIFT HANDLER] Result response: {result.get('response', '')[:100]}...")

                        # Show merged fields if any
                        if result.get("merged_fields"):
                            st.success(f"✅ Updated fields: {', '.join(result['merged_fields'])}")

                        print(f"[DRIFT HANDLER UI] About to render response to user")
                        # Display response
                        st.markdown(result["response"])

                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"]
                        })
                    else:  # reject
                        # User rejected: proceed without applying the conflicting changes
                        st.info("⛔ Keeping original state (language unchanged)")

                        state_reject = agent.get_state()
                        print(f"[DRIFT HANDLER] REJECT - Current state: language={state_reject.tech.language}")

                        current_lang = state_reject.tech.language.value if state_reject.tech.language else "Python"

                        # Generate a response that acknowledges the rejection
                        response = f"""✓ Your original {current_lang.upper()} project settings have been preserved.

I'll continue working with:
- **Language**: {current_lang}
- **Project**: Fish detector with object detection and deep learning
- **Current Setup**: Maintained from your original configuration

Ready to proceed with your {current_lang} implementation!"""

                        st.markdown(response)

                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })

                    # Clean up after handling choice
                    print(f"[DRIFT HANDLER] About to clean up and rerun...")
                    st.session_state.pending_prompt = None
                    st.session_state.pending_result = None
                    st.session_state.drift_choice = None
                    
                    state_before_rerun = agent.get_state()
                    print(f"[DRIFT HANDLER] State BEFORE rerun: language={state_before_rerun.tech.language}")
                    print(f"[DRIFT HANDLER] Cleaned up drift state and calling st.rerun()...")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing your choice: {str(e)}")
                    print(f"[DRIFT HANDLER ERROR] {str(e)}")
                    st.session_state.drift_choice = None
                    st.session_state.pending_prompt = None
                    st.session_state.pending_result = None
                    st.stop()

        else:
            # No choice yet: keep the popup visible and block new chat input.
            print(f"[DRIFT HANDLER] Showing popup, waiting for user choice")
            with st.chat_message("assistant"):
                show_drift_popup(
                    pending_result.get("extraction"),
                    pending_result.get("conflicts", []),
                    agent.get_state()
                )
                st.info("👆 Please choose above: Accept new values or Stick with original")
            st.stop()
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        print(f"\n[CHAT INPUT] New prompt received: '{prompt[:60]}...'")
        print(f"[CHAT INPUT] Current drift state:")
        print(f"  - drift_choice: {st.session_state.get('drift_choice')}")
        print(f"  - pending_prompt: {st.session_state.get('pending_prompt') is not None}")
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process through agent
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    print(f"\n[PROCESS NORMAL] Starting normal process() for: '{prompt[:60]}...'")
                    state_before = agent.get_state()
                    print(f"[PROCESS NORMAL] State BEFORE: language={state_before.tech.language}")
                    
                    result = agent.process(prompt)
                    
                    state_after = agent.get_state()
                    print(f"[PROCESS NORMAL] State AFTER: language={state_after.tech.language}")
                    print(f"[PROCESS NORMAL] Conflicts detected: {len(result.get('conflicts', []))}")
                    
                    # Show drift pop-up if conflicts detected
                    if result.get("conflicts"):
                        # Store pending prompt and result for drift handling
                        st.session_state.pending_prompt = prompt
                        st.session_state.pending_result = result
                        
                        print(f"\n[SET STATE] Storing pending state:")
                        print(f"  - pending_prompt: '{prompt[:60]}...'")
                        print(f"  - pending_result conflicts: {len(result.get('conflicts', []))}")
                        print(f"  - drift_choice: {st.session_state.get('drift_choice')}")
                        
                        drift_container = st.container()
                        with drift_container:
                            show_drift_popup(
                                result.get("extraction"),
                                result.get("conflicts"),
                                agent.get_state()
                            )
                        
                        # Tell user to make a choice
                        st.info("👆 Please choose above: Accept new values or Stick with original")
                        print(f"[SET STATE] Calling st.stop() to wait for user choice...")
                        st.stop()  # Stop processing until user makes a choice
                    
                    # No conflicts - display response normally
                    print(f"[PROCESS] No conflicts detected - displaying response normally")
                    st.markdown(result["response"])
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"]
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Error: {str(e)}"
                    })


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    agent = get_agent()
    render_chat_interface(agent)


if __name__ == "__main__":
    main()
