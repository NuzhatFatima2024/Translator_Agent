from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print(gemini_api_key)
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Write Agent
writer = Agent(
    name = 'Translator Agent',
    instructions= 
    """You are a translator agent. give me english translation of given paragraph مصنوعی ذہانت (Artificial Intelligence) ایک جدید ٹیکنالوجی ہے جو انسانوں کی طرح سوچنے، سیکھنے اور فیصلے کرنے کی صلاحیت رکھتی ہے۔ یہ ٹیکنالوجی کمپیوٹرز اور مشینوں کو اس قابل بناتی ہے کہ وہ بغیر انسانی مدد کے مختلف مسائل کا حل تلاش کریں اور خود سے سیکھتے ہوئے مزید بہتر کارکردگی دکھائیں۔ مصنوعی ذہانت آج کل زندگی کے مختلف شعبوں جیسے صحت، تعلیم، تجارت، زراعت اور سیکیورٹی میں استعمال ہو رہی ہے۔ اس کی مدد سے روزمرہ کے کاموں کو زیادہ آسان اور مؤثر بنایا جا رہا ہے، تاہم اس کے استعمال میں احتیاط بھی ضروری ہے تاکہ اس کے منفی اثرات سے بچا جا سکے۔ ."""
)

response = Runner.run_sync(
    writer,
    input = 'translate a given paragraph in english..',
    run_config = config
    )
print(response.final_output)