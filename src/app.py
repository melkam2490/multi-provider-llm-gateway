
import asyncio

import streamlit as st
from dotenv import load_dotenv
from gateway import UnifiedLLMGateway

load_dotenv()

st.set_page_config(
    page_title="Multi-Provider LLM Gateway",
    page_icon="🤖",
    layout="wide",
)

st.title("Multi-Provider LLM Gateway")
st.write("Compare AI providers through one unified interface.")

provider = st.selectbox(
    "Choose an AI provider",
    ["OpenAI", "Anthropic"],
)

if provider == "OpenAI":
    model = st.selectbox(
        "Choose a model",
        ["gpt-4o-mini", "gpt-4o"],
    )
else:
    model = st.selectbox(
        "Choose a model",
        [
            "claude-3-5-haiku-latest",
            "claude-3-5-sonnet-latest",
        ],
    )

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Ask the AI something...",
)

temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
)

if st.button("Send prompt", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        response_area = st.empty()
        status = st.empty()

        async def run_request():
            gateway = UnifiedLLMGateway()

            try:
                full_response = ""
                final_telemetry = None

                async for chunk in gateway.stream(
                    prompt=prompt.strip(),
                    model=model,
                    temperature=temperature,
                ):
                    if chunk.delta_text:
                        full_response += chunk.delta_text
                        response_area.markdown(full_response)

                    if chunk.is_final:
                        final_telemetry = chunk.telemetry

                return final_telemetry

            finally:
                await gateway.aclose()

        try:
            with st.spinner("Waiting for the AI response..."):
                telemetry = asyncio.run(run_request())

            status.success("Response completed!")

            if telemetry:
                st.subheader("Request telemetry")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "TTFT",
                    f"{telemetry.ttft_ms:.2f} ms",
                )
                col2.metric(
                    "Generation speed",
                    f"{telemetry.tokens_per_second:.2f} tokens/sec",
                )
                col3.metric(
                    "Total tokens",
                    telemetry.total_tokens,
                )

                col4, col5, col6 = st.columns(3)

                col4.metric(
                    "Prompt tokens",
                    telemetry.input_tokens,
                )
                col5.metric(
                    "Completion tokens",
                    telemetry.output_tokens,
                )
                col6.metric(
                    "Estimated cost",
                    f"${telemetry.total_cost_usd:.6f}",
                )

        except Exception as exc:
            print(
                f"Request failed: {type(exc).__name__}: {exc}"
            )
            status.error(
                "The request failed. Check the terminal "
                "for details. Do not share your API keys."
            )
