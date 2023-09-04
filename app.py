import os

import replicate
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for
from flask_wtf import FlaskForm
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

idea_template = """I am a creative children's author with sci/fi technological interests.
I want to write dynamic choose your own adventure stories in which the user chooses between different paths.

Please respond to an initial story prompt by generating 1 page (3-4 paragraphs) of story content that lead up to a decision, to be made by the reader. It will present 2-3 choices.
The reader will make the decision, then the story will continue with 1 more page, and another decision of 2-3 choices.
If the reader enters a different suggestion, follow that instead.

Start generating the first page for the story, followed by the choices.

The story idea is:
{idea}

Page 1:
"""

prompt = PromptTemplate(template=idea_template, input_variables=["idea"])


app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"  # Replace with your secret key

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
REPLICATE_MODEL = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
# REPLICATE_MODEL = (
#     "stability-ai/sdxl:d830ba5dabf8090ec0db6c10fc862c6eb1c929e1a194a5411852d25fd954ac82"
# )

llm = OpenAI()
llm_chain = LLMChain(prompt=prompt, llm=llm)


class StoryIdeaForm(FlaskForm):
    idea = StringField(
        "What should this story be about?",
        validators=[DataRequired()],
        render_kw={"class": "form-control"},
    )
    submit = SubmitField("Submit Idea", render_kw={"class": "btn btn-primary"})


class PromptForm(FlaskForm):
    prompt = StringField()


class ChatForm(FlaskForm):
    choice = StringField(
        "What should happen next?",
        validators=[DataRequired()],
        render_kw={"class": "form-control"},
    )
    submit = SubmitField("Submit", render_kw={"class": "btn btn-primary"})


@app.route("/", methods=["GET", "POST"])
def submit_idea():
    form = StoryIdeaForm()

    if form.validate_on_submit():
        idea = form.idea.data
        return redirect(url_for("generate_story", idea=idea))

    return render_template("submit_idea.html", form=form)


@app.route("/generate_image", methods=["GET", "POST"])
def generate_image():
    form = PromptForm()
    prompt = form.prompt.data
    full_prompt = f"black and white children's book pen illustration of {prompt}"
    image_url = replicate.run(
        REPLICATE_MODEL,
        input={"prompt": full_prompt},
    )[0]
    return render_template(
        "fragments/img_response.html",
        image_url=image_url,
        image_prompt=prompt,
    )


@app.route("/generate_text", methods=["GET", "POST"])
def generate_text():
    form = PromptForm()
    prompt = form.prompt.data

    # Get LLM response with LangChain OpenAI
    llm_response = llm_chain.run(prompt)

    return render_template(
        "fragments/llm_response.html",
        llm_response=llm_response.strip(),
        idea=prompt,
    )


@app.route("/generate_story", methods=["GET", "POST"])
def generate_story():
    form = ChatForm()

    if form.validate_on_submit():
        choice = form.choice.data

        return render_template(
            "generate_story.html",
            # The Idea is a query parameter that is passed to the page
            form=form,
            idea=request.args.get("idea"),
            prompt=choice,
        )

    return render_template(
        "generate_story.html",
        form=form,
        idea=request.args.get("idea"),
        prompt=request.args.get("idea"),
    )


if __name__ == "__main__":
    app.run(debug=True)
