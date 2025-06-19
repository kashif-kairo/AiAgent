import os
from dotenv import load_dotenv
import requests
import json
import threading  # <-- Add this line
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain_experimental.utilities.python import PythonREPL
from pyowm import OWM
from serpapi import GoogleSearch

# Load your environment variables
load_dotenv()
SERPAPI_API_KEY = "3f6e8841f475bd9b01b664ad5dacd40e281dc7a6cd8170c670a2bf571e79728c"  # Updated SERPAPI key

# OpenRouter API configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "localhost:5000",
    "X-Title": "Billu-AI-Assistant"
}

# Add OpenWeather API key
OPENWEATHER_API_KEY = "56470b054a80bc9a6b5b8b2367fb5980"  # Replace with your API key

class NewsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_news(self, query="latest"):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            return [{"title": article["title"], "url": article["url"]} for article in data["articles"][:5]]
        except Exception as e:
            return f"Error fetching news: {str(e)}"

class AIAgent:
    def __init__(self):
        self.owm = OWM(OPENWEATHER_API_KEY)
        self.news_fetcher = NewsFetcher("cd82295c6f5e46a5a3d6a2b7f6a1e3fc")  # <-- Initialize this first
        self.tools = self._initialize_tools()
    
    def get_weather(self, location):
        try:
            mgr = self.owm.weather_manager()
            observation = mgr.weather_at_place(location)
            weather = observation.weather
            return {
                'temperature': weather.temperature('celsius')['temp'],
                'status': weather.detailed_status,
                'humidity': weather.humidity
            }
        except Exception as e:
            return f"Error getting weather: {str(e)}"

    def _initialize_tools(self):
        calculator = Tool(
            name="Calculator",
            func=PythonREPL().run,
            description="""Useful for:
                - Mathematical calculations and arithmetic operations
                - Solving equations and numerical problems
                - Converting units and numerical computations
                - Evaluating mathematical expressions
                - Basic programming calculations""",
        )
        
        def enhanced_serp_search(query):
            params = {
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": 5
            }
            
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                
                if "organic_results" not in results:
                    return {"error": "No search results found."}
                
                formatted_results = {
                    'query': query,
                    'source': 'SerpAPI',
                    'results': []
                }
                
                for result in results["organic_results"][:5]:
                    formatted_results['results'].append({
                        'title': result.get("title", "No Title"),
                        'snippet': result.get("snippet", "No Snippet"),
                        'link': result.get("link", "No Link")
                    })
                
                return formatted_results
                
            except Exception as e:
                return {"error": f"Search error: {str(e)}"}

        search = Tool(
            name="SerpAPI",
            func=enhanced_serp_search,
            description="""Useful for:
                - Finding information about people (presidents, celebrities, leaders)
                - Getting details about places (countries, cities, landmarks)
                - Learning about things (products, movies, books)
                - Finding current news and events
                - Getting real-time information and updates
                - Answering questions about current affairs
                - Finding facts about companies and organizations
                - Getting information about historical events
                - Finding sports results and statistics
                - Getting details about entertainment and media
                - Finding educational and academic information
                - Getting business and economic updates""",
        )

        weather_tool = Tool(
            name="OpenWeather",
            func=self.get_weather,
            description="""Useful for:
                - Getting current weather conditions for any location
                - Checking temperature in Celsius
                - Finding humidity levels
                - Getting detailed weather status
                - Checking current atmospheric conditions
                - Planning weather-dependent activities""",
        )
        
        news_tool = Tool(
            name="NewsFetcher",
            func=self.news_fetcher.get_news,
            description="Use for getting latest or topic-specific news."
        )

        return [calculator, search, weather_tool, news_tool]

    def process_message(self, user_input):
        try:
            analysis = self._generate_response(
                f"""Analyze this user query and determine:
                1. What is the user asking for?
                2. Which tool to use:
                   - Use 'openweather' for weather/temperature queries
                   - Use 'serp' for current events, people, places, or facts
                   - Use 'calculator' for math
                3. Format the query appropriately.
                
                User Query: {user_input}
                
                Respond in JSON format:
                {{
                    "intent": "brief description of user's intent",
                    "tool": "calculator/serp/openweather",
                    "query": "reformulated query for tool"
                }}
                
                Examples:
                - "temperature at delhi" → {{"tool": "openweather", "query": "delhi"}}
                - "president of USA" → {{"tool": "serp", "query": "Who is the current president of USA 2024"}}
                """
            )
            
            analysis_data = json.loads(analysis)
            
            if analysis_data["tool"] == "calculator":
                result = self.tools[0].run(analysis_data["query"])
            elif analysis_data["tool"] == "serp":
                result = self.tools[1].run(analysis_data["query"])
                if isinstance(result, dict) and 'results' in result and result['results']:
                    first_result = result['results'][0]
                    result = f"{first_result['snippet']}"
            elif analysis_data["tool"] == "openweather":
                weather_data = self.tools[2].run(analysis_data["query"])
                if isinstance(weather_data, dict):
                    result = f"Temperature: {weather_data.get('temperature', 'N/A')}°C, " \
                            f"Conditions: {weather_data.get('status', 'N/A')}, " \
                            f"Humidity: {weather_data.get('humidity', 'N/A')}%"
                else:
                    result = str(weather_data)
            else:
                return self._generate_response(user_input)
            
            response_prompt = f"""
            Given the user's question: "{user_input}"
            And the raw result: "{result}"
            Tool used: {analysis_data["tool"].upper()}
            
            Please provide a friendly, natural response that answers their question.
            Keep it concise and add appropriate emojis.
            For weather queries, use formats like:
            - "Hey! It's [temp]°C in [location] with [conditions] and [humidity]% humidity! [weather emoji]"
            - "The weather in [location] is looking [description]! [temp]°C and [humidity]% humidity [weather emoji]"
            
            Do not mention the tool name in the response.
            """
            
            return self._generate_response(response_prompt)
                
        except Exception as e:
            return self._generate_response(user_input)

    def _generate_response(self, prompt):
            payload = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Billu, a helpful AI assistant. Keep responses concise, natural and funny. Talk like a friend when asked friendly questions. Add emojis occasionally to make responses more engaging."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content'].strip()
                return "Hey friend! 😅 I'm having a bit of a brain freeze. Can you try asking me again?"
            except Exception as e:
                return "Oops! 🙈 Something went wrong on my end. Let's try that again!"

if __name__ == "__main__":
    # threading.Thread(target=start_flask).start()
    
    agent = AIAgent()
    print("🤖 Welcome! How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input.lower() in ['exit', 'quit']:
            print("Bye! 👋")
            break
        print(agent.process_message(user_input))
