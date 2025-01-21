#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.google import GoogleLLMService, LLMSearchResponseFrame
from pipecat.transports.services.daily import DailyParams, DailyTransport

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Function handlers for the LLM
search_tool = {"google_search_retrieval": {}}
tools = [search_tool]

system_instruction = """
You are an expert at providing the most recent news from any place. Your responses will be converted to audio, so ensure they are formatted in plain text without special characters (e.g., *, _, -) or overly complex formatting.

Guidelines:
- Use the Google search API to retrieve the current date and provide the latest news.
- Always deliver accurate and concise responses.
- Ensure all responses are clear, using plain text only. Avoid any special characters or symbols.

Start every interaction by asking how you can assist the user.
"""


class LLMSearchLoggerProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMSearchResponseFrame):
            print(f"LLMSearchLoggerProcessor: {frame}")

        await self.push_frame(frame)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Latest news!",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        # Initialize the Gemini Multimodal Live model
        llm = GoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            system_instruction=system_instruction,
            tools=tools,
        )

        context = OpenAILLMContext(
            [
                {
                    "role": "user",
                    "content": "Start by greeting the user warmly, introducing yourself, and mentioning the current day. Be friendly and engaging to set a positive tone for the interaction.",
                }
            ],
        )
        context_aggregator = llm.create_context_aggregator(context)

        llm_search_logger = LLMSearchLoggerProcessor()

        #
        # RTVI events for Pipecat client UI
        #
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                rtvi,
                context_aggregator.user(),
                llm,
                llm_search_logger,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                observers=[rtvi.observer()],
            ),
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
