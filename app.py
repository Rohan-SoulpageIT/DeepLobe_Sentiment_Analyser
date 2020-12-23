from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
import uvicorn
from starlette.templating import Jinja2Templates
from starlette.config import Config
from flair.models import TextClassifier
from flair.data import Sentence
import time

classifier = TextClassifier.load("en-sentiment")

# getting all the templets for the following dir.
templates = Jinja2Templates(directory="templates")


async def predict(request):
    if request.method == "POST":
        form = await request.form()
        inputQuery = form["message"]
        start = time.time()
        sentence = Sentence(inputQuery)
        classifier.predict(sentence)
        label = sentence.labels[0]
        result = str(label.value)
        context = {
            "Sentiment": result,
            "time (seconds)": str(round(time.time() - start, 3)),
        }
        return JSONResponse(context)
    else:
        return templates.TemplateResponse("home.html", {"request": request, "data": ""})


# All the routs of this website
routes = [
    Route("/upload-deeplobe-sentiment-detector", predict, methods=["GET", "POST"]),
]
# App congiguration.
app = Starlette(
    debug=True,
    routes=routes,
)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)