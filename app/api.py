from fastapi import APIRouter
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


@router.get("/extract-topics")
@limiter.limit("30/minute")
def extract_topics(request: Request, response: Response, text: str = None):
    if text is None:
        response.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        return {'Error': 'No text specified'}
    else:
        topics = core.extract_topics(text)
        response.status_code = status.HTTP_200_OK
        return {"topics": topics}
