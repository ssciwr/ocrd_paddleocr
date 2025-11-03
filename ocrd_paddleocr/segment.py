from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd.processor.ocrd_page_result import OcrdPageResult
from ocrd_models.ocrd_page import CoordsType, OcrdPage, TextRegionType, ImageRegionType
from ocrd_utils import points_from_bbox
from paddleocr import LayoutDetection
from typing import Optional

import click
import numpy as np
import pathlib

# Examples for segment models:
#
# * https://github.com/OCR-D/ocrd_kraken/blob/master/ocrd_kraken/segment.py#L63
# * https://github.com/qurator-spk/eynollah/blob/main/src/eynollah/processor.py#L43

#
# Mapping the labels found in PaddleOCR JSON to required PAGE XML information.
# Values are stored as tuples of (class type, class name, subtype).
#
# Building this incrementally as we go...
#
paddleocr_label_to_pagexml_type = {
    "image": (ImageRegionType, "ImageRegion", None),
    "text": (TextRegionType, "TextRegion", None),
    "paragraph_title": (TextRegionType, "TextRegion", "heading"),
    "header": (TextRegionType, "TextRegion", "heading"),
    "table": (TextRegionType, "TextRegion", None),
    "number": (TextRegionType, "TextRegion", None),
    "doc_title": (TextRegionType, "TextRegion", "heading"),
    "figure_title": (TextRegionType, "TextRegion", "heading"),
    "aside_text": (TextRegionType, "TextRegion", None),
    "seal": (ImageRegionType, "ImageRegion", None),
    "footer": (TextRegionType, "TextRegion", None),
}


class PaddleOCRProcessor(Processor):

    def setup(self) -> None:
        #
        # Additional parameters to maybe expose in the future:
        #
        # * layout_unclip_ratio
        # * layout_merge_bboxes_mode
        #
        self.detector = LayoutDetection(
            model_name="PP-DocLayout_plus-L",
            model_dir=pathlib.Path(self.resolve_resource("PP-DocLayout_plus-L")).parent,
            threshold=self.parameter["threshold"],
            layout_nms=self.parameter["layout_nms"],
            layout_merge_bboxes_mode=self.parameter["layout_merge_bboxes_mode"],
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

        output = self.detector.predict(np.array(page_image), batch_size=1)[0]

        for i, box in enumerate(output.json["res"]["boxes"]):
            label = box["label"]

            if label not in paddleocr_label_to_pagexml_type:
                raise ValueError(f"Unknown PaddleOCR label: {label}")

            class_type, class_name, subtype = paddleocr_label_to_pagexml_type[label]

            region = class_type(
                id=f"region_{i+1:04d}",
                type_=subtype,
                Coords=CoordsType(points=points_from_bbox(*box["coordinate"])),
            )

            # Add the region to the PAGE XML structure
            getattr(page, f"add_{class_name}")(region)  # e.g. page.add_TextRegion()

        return result


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(PaddleOCRProcessor, *args, **kwargs)


if __name__ == "__main__":
    cli()
