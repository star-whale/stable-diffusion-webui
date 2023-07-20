import sys
sys.argv=sys.argv[:1] # webui parses system args when importing it
sys.argv.append('--listen')
import os
from webui import webui
from starwhale import handler, Dataset, pass_context, Context
from starwhale.api import model, experiment
from finetune_text_to_image_lora import fine_tune

@handler(expose=7860)
def sdui():
    os.environ["COVERAGE_RUN"]="1" # run webui under COVERAGE_RUN mode
    webui()


@pass_context
@experiment.fine_tune()
def ft(context: Context):
    fine_tune(context)  