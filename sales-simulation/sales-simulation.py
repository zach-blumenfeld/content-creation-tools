import getpass
import json

import openai
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain_community.tools import TavilySearchResults

# Load environment variables from .env file
load_dotenv()
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

# Define the Customer agent
class Customer:
    def __init__(self, initial_prompt):
        self.messages = [HumanMessage(content=initial_prompt)]

    def respond(self, response):
        self.messages.append(AIMessage(content=response))
        return self.messages[-1].content

    def should_continue(self):
        decision_prompt = "Based on the current discussions, should the customer continue asking for more information or have they gathered enough to make a decision? Reply with 'continue' or 'stop'."
        decision_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a customer evaluating AI solutions."},
                {"role": "user", "content": decision_prompt}
            ]
        )
        decision = decision_response['choices'][0]['message']['content'].strip().lower()
        return decision == "continue"

# Define the Product Marketer and Architect agent
class ProductMarketerArchitect:
    def __init__(self, product_name, initial_prompt):
        self.product_name = product_name
        self.messages = [SystemMessage(content=f"You are a product marketer for {product_name}.")]
        self.messages.append(HumanMessage(content=initial_prompt))
        self.model = ChatOpenAI(model_name="gpt-4o")
        self.sales_playbook_path = f"{product_name.replace(' ', '_')}_Sales_Playbook.md"
        self.changelog_path = f"{product_name.replace(' ', '_')}_Changelog.md"
        self.search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            # include_domains=[...],
            # exclude_domains=[...],
            # name="...",            # overwrite default tool name
            # description="...",     # overwrite default tool description
            # args_schema=...,       # overwrite default args_schema: BaseModel
        )
        self.ensure_sales_playbook()

    def ensure_sales_playbook(self):
        if not os.path.exists(self.sales_playbook_path):
            with open(self.sales_playbook_path, "w", encoding="utf-8") as file:
                file.write(f"# {self.product_name} Sales Playbook\n\n")
        if not os.path.exists(self.changelog_path):
            with open(self.changelog_path, "w", encoding="utf-8") as file:
                file.write(f"# {self.product_name} Sales Playbook Changelog\n\n")

    def web_search(self, query):
        search_results = self.search_tool.run(query)
        return json.dumps(search_results[:5], indent=2)  # Convert the first 5 results to a formatted JSON string

    def update_sales_playbook(self, customer_message, response):
        update_prompt = f"Based on the customer question and your response, suggest updates to the sales playbook.\n\nCustomer Message: {customer_message}\n\nYour Response: {response}\n\n"
        update_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a strategic sales expert updating a sales playbook."},
                {"role": "user", "content": update_prompt}
            ]
        )
        update_text = update_response['choices'][0]['message']['content']

        with open(self.sales_playbook_path, "a", encoding="utf-8") as file:
            file.write(f"{update_text}\n\n")

        with open(self.changelog_path, "a", encoding="utf-8") as file:
            file.write(f"Updated sales playbook based on new customer conversation.\n\n{update_text}\n\n")

    def converse(self, customer_message):
        search_query = f"{self.product_name} for AI agentic systems"
        additional_info = self.web_search(search_query)

        self.messages.append(HumanMessage(content=customer_message))
        response = self.model.invoke(self.messages + [HumanMessage(content=f"Here is additional information from the web:\n{additional_info}")])
        self.messages.append(response)

        self.update_sales_playbook(customer_message, response.content)

        return response.content

# Conversation simulation
customer_prompt = "I am evaluating a solution for my Agentic AI system. Convince me why your approach is best."
customer = Customer(customer_prompt)

competitors = [
    ProductMarketerArchitect("Semantic Layer Co.", "Explain why a semantic layer is essential for AI agentic systems."),
    ProductMarketerArchitect("GraphDB Inc.", "Explain why a graph database is the superior choice for AI agentic systems."),
]

conversation_log = "# AI Agent Evaluation Conversations\n\n"
iteration = 1

while True:
    conversation_log += f"## Iteration {iteration}\n\n"
    for competitor in competitors:
        conversation_log += f"### {competitor.product_name}\n\n"
        conversation_log += f"**Customer:** {customer.messages[-1].content}\n\n"
        response = competitor.converse(customer.messages[-1].content)
        conversation_log += f"**{competitor.product_name}:** {response}\n\n"

        # Customer refines objections
        objection_prompt = f"Given the following response, what would be a strong objection or counterpoint from a skeptical customer?\n\n{response}"
        objection_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a skeptical customer analyzing the response."},
                {"role": "user", "content": objection_prompt}
            ]
        )

        objection = objection_response['choices'][0]['message']['content']
        conversation_log += f"**Customer Objection:** {objection}\n\n"
        customer.respond(response + "\n\n" + objection)

    if not customer.should_continue():
        break
    iteration += 1

# Generate final decision
final_decision_prompt = f"Summarize the key points from these conversations and determine which solution best fits the customer’s needs. Justify the final choice.\n\n{conversation_log}"
final_decision_response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an impartial analyst evaluating the conversation."},
        {"role": "user", "content": final_decision_prompt}
    ]
)

final_decision = final_decision_response['choices'][0]['message']['content']
conversation_log += f"# Final Decision\n\n{final_decision}\n"

# Save conversation log to markdown file
with open("conversation_log.md", "w", encoding="utf-8") as file:
    file.write(conversation_log)

print("Conversation log saved to conversation_log.md")
