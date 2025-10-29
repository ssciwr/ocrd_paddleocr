from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd.processor.ocrd_page_result import OcrdPageResult
from ocrd_models.ocrd_page import OcrdPage
from paddleocr import LayoutDetection
from typing import Optional

import click
import pathlib

# Examples for segment models:
#
# * https://github.com/OCR-D/ocrd_kraken/blob/master/ocrd_kraken/segment.py#L63
# * https://github.com/qurator-spk/eynollah/blob/main/src/eynollah/processor.py#L43


class PaddleOCRProcessor(Processor):

    def setup(self) -> None:
        #
        # Parameters to expose in the future
        #
        # * threshold
        # * layout_nms
        # * layout_unclip_ratio
        # * layout_merge_bboxes_mode
        #
        self.detector = LayoutDetection(
            model_name="PP-DocLayout_plus-L",
            model_dir=pathlib.Path(self.resolve_resource("PP-DocLayout_plus-L")).parent,
        )

    def shutdown(self) -> None:
        pass

    def process_page_pcgts(
        self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None
    ) -> OcrdPageResult:
        # Most of this code is copied from eynollah, as it would require understanding of OCR-D internals:
        # https://github.com/qurator-spk/eynollah/blob/main/src/eynollah/processor.py#L43

        assert input_pcgts
        assert input_pcgts[0]
        assert self.parameter
        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()

        # Here, it very unclear what the feature_filter does in detail. Eynollah
        # states 'cropped,deskewed,binarized' but I would have assumed that the
        # decision is with the user whether to apply segmentation to binarized images
        # or not. How does this even interact with the input file group then?
        page_image, page_coords, page_info = self.workspace.image_from_page(
            page, page_id
        )

        output = self.detector.predict(page_image, batch_size=1)[0]

        return result


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(PaddleOCRProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
