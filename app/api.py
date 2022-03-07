from typing import List

from fastapi import APIRouter
from fastapi import Query
from starlette import status
from starlette.requests import Request
from starlette.responses import Response

from services.core import Core
from utils.logger import Logger
from utils.requestlimiter import RequestLimiter

router = APIRouter()
core = Core()
logger = Logger('Api')
request_limiter = RequestLimiter.instance()
limiter = request_limiter.limiter


@router.get("/healthcheck")
def healthcheck():
    return {"Status": "Alive"}


@router.get("/extract-keywords")
@limiter.limit("30/minute")
async def extract_keywords(request: Request, response: Response, text: str = None, top_n: int = 10):
    body = await request.json()

    if 'text' in body and body['text'] is not None:
        text = body['text']
    if 'top_n' in body and body['top_n'] is not None:
        top_n = body['top_n']
    if text is None:
        response.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        return {'Error': 'No text specified'}
    else:
        keywords = core.extract_keywords(text, top_n)
        response.status_code = status.HTTP_200_OK
        return {"keywords": keywords}


@router.get("/classify-text-labels")
@limiter.limit("30/minute")
async def classify_text_labels(request: Request, response: Response,
                               text: str = None,
                               labels: List[str] = Query(None),
                               multi: bool = True):
    body = await request.json()

    if 'text' in body and body['text'] is not None:
        text = body['text']
    if 'labels' in body and body['labels'] is not None:
        labels = body['labels']
    if 'multi' in body and body['multi'] is not None:
        multi = body['multi']
    if text is None:
        response.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        return {'Error': 'No text specified'}
    else:
        labels = core.classify_text_labels(text, labels, multi)
        response.status_code = status.HTTP_200_OK
        return {"labels": labels}
