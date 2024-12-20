import sys
import os

# Add the parent directory of colpali_engine to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image 

from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali import ColPali
#from colpali_engine.models.paligemma_colbert_architecture import ColPali
#  from colpali_engine.trainer.colmodel_training import CustomRetrievalEvaluator
#from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator as CustomRetrievalEvaluator
from colpali_engine.utils.torch_utils import process_images, process_queries
from colpali_engine.utils.dataset_transformation import load_from_dataset
#from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator  


def main() -> None:
    """Example script to run inference with ColPali"""

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     if torch.cuda.is_bf16_supported():
    #         type = torch.bfloat16
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     type = torch.float32
    # else:
    #     device = torch.device("cpu")
    #     type = torch.float32
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        type = torch.float32
    else:
        device = torch.device("cpu")
        type = torch.float32

    # Load model
    model_name = "vidore/colpali"
    model = ColPali.from_pretrained("vidore/colpaligemma-3b-pt-448-base", torch_dtype=type).eval()
    model.load_adapter(model_name)
    model = model.eval()
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    print('model loaded')

    # select images -> load_from_pdf(<pdf_path>),  load_from_image_urls(["<url_1>"]), load_from_dataset(<path>)
    images = load_from_dataset("vidore/docvqa_test_subsampled")
    queries = ["From which university does James V. Fiorca come ?", "Who is the japanese prime minister?"]

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    print('load data here')
    ds = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc)))

    # run inference - queries
    dataloader = DataLoader(
        queries,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_queries(processor, x, Image.new("RGB", (448, 448), (255, 255, 255))),
    )

    qs = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query)))

    # run evaluation
    # retriever_evaluator = CustomRetrievalEvaluator(is_multi_vector=True)
    # scores = retriever_evaluator.evaluate(qs, ds)
    # print(scores.argmax(axis=1))


if __name__ == "__main__":
    typer.run(main)