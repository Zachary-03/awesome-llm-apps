import asyncio
import os
import re
import streamlit as st
from typing import Dict, Any, List, Tuple
from openai import AsyncOpenAI
from agents import Agent, Runner, set_default_openai_client, set_default_openai_api
from firecrawl import FirecrawlApp
from agents.tool import function_tool
from agents.run import RunConfig

# MiniMax OpenAI 兼容 API
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_MODEL = "abab6.5s-chat"

def load_secrets(secrets_path: str | None = None) -> Tuple[str, str]:
    """从 secrets.md 读取 Firecrawl 和 MiniMax API key。返回 (firecrawl_key, minimax_key)。"""
    path = secrets_path or os.environ.get("SECRETS_FILE", os.path.expanduser("~/secrets.md"))
    firecrawl_key = ""
    minimax_key = ""
    if not os.path.isfile(path):
        return firecrawl_key, minimax_key
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Firecrawl：xxx 或 Firecrawl: xxx（全角/半角冒号）
        m = re.match(r"Firecrawl\s*[：:]\s*(.+)", line, re.IGNORECASE)
        if m:
            firecrawl_key = m.group(1).strip()
            i += 1
            continue
        if re.match(r"minimax\s*[：:]", line, re.IGNORECASE):
            if i + 1 < len(lines):
                minimax_key = lines[i + 1].strip()
            i += 2
            continue
        i += 1
    return firecrawl_key, minimax_key

# 从 secrets.md 加载默认 key
_default_firecrawl, _default_minimax = load_secrets()

# Set page configuration
st.set_page_config(
    page_title="Deep Research Agent (MiniMax)",
    page_icon="📘",
    layout="wide"
)

# Initialize session state for API keys (优先使用 secrets 中的默认值)
if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = _default_firecrawl
if "minimax_api_key" not in st.session_state:
    st.session_state.minimax_api_key = _default_minimax

# Sidebar for API keys
with st.sidebar:
    st.title("API Configuration")
    st.caption("Firecrawl 与 MiniMax 可从 ~/secrets.md 自动加载")
    firecrawl_api_key = st.text_input(
        "Firecrawl API Key",
        value=st.session_state.firecrawl_api_key,
        type="password"
    )
    minimax_api_key = st.text_input(
        "MiniMax API Key",
        value=st.session_state.minimax_api_key,
        type="password"
    )

    if firecrawl_api_key:
        st.session_state.firecrawl_api_key = firecrawl_api_key
    if minimax_api_key:
        st.session_state.minimax_api_key = minimax_api_key
        # 使用 MiniMax（OpenAI 兼容接口）
        set_default_openai_api("chat_completions")
        set_default_openai_client(
            AsyncOpenAI(base_url=MINIMAX_BASE_URL, api_key=minimax_api_key)
        )

# Main content
st.title("📘 Deep Research Agent (MiniMax)")
st.markdown("基于 MiniMax 与 Firecrawl 的深度研究助手，可从 ~/secrets.md 读取 API 配置")

# Research topic input
research_topic = st.text_input("Enter your research topic:", placeholder="e.g., Latest developments in AI")

# Keep the original deep_research tool
@function_tool
async def deep_research(query: str, max_depth: int, time_limit: int, max_urls: int) -> Dict[str, Any]:
    """
    Perform comprehensive web research using Firecrawl's deep research endpoint.
    """
    try:
        # Initialize FirecrawlApp with the API key from session state
        firecrawl_app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
        
        # Define research parameters
        params = {
            "maxDepth": max_depth,
            "timeLimit": time_limit,
            "maxUrls": max_urls
        }
        
        # Set up a callback for real-time updates
        def on_activity(activity):
            st.write(f"[{activity['type']}] {activity['message']}")
        
        # Run deep research
        with st.spinner("Performing deep research..."):
            results = firecrawl_app.deep_research(
                query=query,
                params=params,
                on_activity=on_activity
            )
        
        return {
            "success": True,
            "final_analysis": results['data']['finalAnalysis'],
            "sources_count": len(results['data']['sources']),
            "sources": results['data']['sources']
        }
    except Exception as e:
        st.error(f"Deep research error: {str(e)}")
        return {"error": str(e), "success": False}

# Keep the original agents
research_agent = Agent(
    name="research_agent",
    instructions="""You are a research assistant that can perform deep web research on any topic.

    When given a research topic or question:
    1. Use the deep_research tool to gather comprehensive information
       - Always use these parameters:
         * max_depth: 3 (for moderate depth)
         * time_limit: 180 (3 minutes)
         * max_urls: 10 (sufficient sources)
    2. The tool will search the web, analyze multiple sources, and provide a synthesis
    3. Review the research results and organize them into a well-structured report
    4. Include proper citations for all sources
    5. Highlight key findings and insights
    """,
    tools=[deep_research]
)

elaboration_agent = Agent(
    name="elaboration_agent",
    instructions="""You are an expert content enhancer specializing in research elaboration.

    When given a research report:
    1. Analyze the structure and content of the report
    2. Enhance the report by:
       - Adding more detailed explanations of complex concepts
       - Including relevant examples, case studies, and real-world applications
       - Expanding on key points with additional context and nuance
       - Adding visual elements descriptions (charts, diagrams, infographics)
       - Incorporating latest trends and future predictions
       - Suggesting practical implications for different stakeholders
    3. Maintain academic rigor and factual accuracy
    4. Preserve the original structure while making it more comprehensive
    5. Ensure all additions are relevant and valuable to the topic
    """
)

_run_config = RunConfig(model=MINIMAX_MODEL)

async def run_research_process(topic: str):
    """Run the complete research process."""
    # Step 1: Initial Research
    with st.spinner("Conducting initial research..."):
        research_result = await Runner.run(research_agent, topic, run_config=_run_config)
        initial_report = research_result.final_output

    # Display initial report in an expander
    with st.expander("View Initial Research Report"):
        st.markdown(initial_report)

    # Step 2: Enhance the report
    with st.spinner("Enhancing the report with additional information..."):
        elaboration_input = f"""
        RESEARCH TOPIC: {topic}

        INITIAL RESEARCH REPORT:
        {initial_report}

        Please enhance this research report with additional information, examples, case studies,
        and deeper insights while maintaining its academic rigor and factual accuracy.
        """

        elaboration_result = await Runner.run(elaboration_agent, elaboration_input, run_config=_run_config)
        enhanced_report = elaboration_result.final_output

    return enhanced_report

# Main research process
if st.button("Start Research", disabled=not (minimax_api_key and firecrawl_api_key and research_topic)):
    if not minimax_api_key or not firecrawl_api_key:
        st.warning("请在侧栏填写 Firecrawl 与 MiniMax API Key，或确保 ~/secrets.md 中已配置。")
    elif not research_topic:
        st.warning("请输入研究主题。")
    else:
        try:
            # Create placeholder for the final report
            report_placeholder = st.empty()
            
            # Run the research process
            enhanced_report = asyncio.run(run_research_process(research_topic))
            
            # Display the enhanced report
            report_placeholder.markdown("## Enhanced Research Report")
            report_placeholder.markdown(enhanced_report)
            
            # Add download button
            st.download_button(
                "Download Report",
                enhanced_report,
                file_name=f"{research_topic.replace(' ', '_')}_report.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Powered by MiniMax + Firecrawl · API 从 secrets.md 加载") 