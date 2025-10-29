from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd.processor.ocrd_page_result import OcrdPageResult
from ocrd_models.ocrd_page import OcrdPage
from typing import Optional

import click


class PaddleOCRProcessor(Processor):
    @property
    def executable(self) -> str:
        return "ocrd-paddleocr-segment"

    def setup(self) -> None:
        model = self.resolve_resource("PP-DocLayout_plus-L")

    def shutdown(self) -> None:
        pass

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        return OcrdPageResult()


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(PaddleOCRProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
