#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from processors import StoryImageProcessor, StoryProcessor
from prompts import CUE_USER_TURN, LLM_BASE_PROMPT
from utils.helpers import load_images, load_sounds

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, StopTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.logger import FrameLogger
from pipecat.services.cartesia import CartesiaHttpTTSService, CartesiaTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.fal import FalImageGenService
from pipecat.services.google import GoogleLLMService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
    DailyTransportMessageFrame,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sounds = load_sounds(["listening.wav"])
images = load_images(["book1.png", "book2.png"])


async def main(room_url, token=None):
    async with aiohttp.ClientSession() as session:
        # -------------- Transport --------------- #

        transport = DailyTransport(
            room_url,
            token,
            "Storytelling Bot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=768,
                camera_out_height=768,
                transcription_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_enabled=True,
            ),
        )

        logger.debug("Transport created for room:" + room_url)

        # -------------- Services --------------- #

        llm_service = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))
        # llm_service = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # tts_service = CartesiaTTSService(
        #     api_key=os.getenv("CARTESIA_API_KEY"),
        #     voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        # )
        tts_service = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id=os.getenv("ELEVENLABS_VOICE_ID")
        )

        fal_service_params = FalImageGenService.InputParams(
            image_size={"width": 768, "height": 768}
        )

        fal_service = FalImageGenService(
            aiohttp_session=session,
            # model="fal-ai/fast-lightning-sdxl",
            model="fal-ai/stable-diffusion-v35-medium",
            params=fal_service_params,
            key=os.getenv("FAL_KEY"),
        )

        # --------------- Setup ----------------- #

        message_history = [LLM_BASE_PROMPT]
        story_pages = []

        # We need aggregators to keep track of user and LLM responses
        context = OpenAILLMContext(message_history)
        context_aggregator = llm_service.create_context_aggregator(context)

        # -------------- Processors ------------- #

        story_processor = StoryProcessor(message_history, story_pages)
        image_processor = StoryImageProcessor(fal_service)

        # -------------- Story Loop ------------- #

        runner = PipelineRunner()

        # The intro pipeline is used to start
        # the story (as per LLM_INTRO_PROMPT)
        fl2 = FrameLogger("intro pipeline")

        logger.debug("Waiting for participant...")

        # We run the intro pipeline. This will start the transport. The intro
        # task will exit after StopTaskFrame is processed.

        # The main story pipeline is used to continue the story based on user
        # input.
        fl = FrameLogger("after image processor", "green")
        fl2 = FrameLogger("after story processor", "red")
        fl3 = FrameLogger("After transport output", "cyan")
        fl4 = FrameLogger("Before transport output", "magenta")
        main_pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm_service,
                story_processor,
                fl2,
                image_processor,
                fl,
                tts_service,
                fl4,
                transport.output(),
                fl3,
                context_aggregator.assistant(),
            ]
        )

        main_task = PipelineTask(main_pipeline)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.debug("Participant joined, storytime commence!")
            await transport.capture_participant_transcription(participant["id"])
            await main_task.queue_frames(
                [
                    images["book1"],
                    context_aggregator.user().get_context_frame(),
                    DailyTransportMessageFrame(CUE_USER_TURN),
                    # sounds["listening"],
                    images["book2"],
                ]
            )

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await main_task.queue_frame(EndFrame())

        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            if state == "left":
                await main_task.queue_frame(EndFrame())

        await runner.run(main_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Storyteller Bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t))
