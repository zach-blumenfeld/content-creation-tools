import getpass
import json
import datetime
from typing import List

import openai
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.tools import TavilySearchResults
from pydantic import BaseModel, Field
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")


class Queries(BaseModel):
    queries: List[str] = Field(description="list of queries strings to run")


# Define the Customer agent
class Customer:
    def __init__(self, initial_prompt, competing_vendors):
        self.initial_prompt = initial_prompt
        self.messages = {cv: [HumanMessage(content=initial_prompt)] for cv in competing_vendors}
        self.model = ChatOpenAI(model_name="gpt-4o")
        self.log_file = "simulated-content/customer-log.md"

        # Ensure log file exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as file:
                file.write("# Customer Interaction Log\n\n")

    def _make_objections(self, vendor_to_object):
        """
        Develop objections for the specified vendor based on responses from other vendors in the last round.
        """
        context = "\n\n".join(
            [f"Vendor: {vendor}\nResponse: {self.messages[vendor][-1].content}" for vendor in self.messages if
             vendor != vendor_to_object]
        )
        history = '\n'.join([m.content for m in self.messages[vendor_to_object]])
        objection_prompt = (f"Based on the below discussions with {vendor_to_object} and the context from competing vendors, generate strong objections for {vendor_to_object}:"
                            f"\n\n# Discussions with {vendor_to_object}\n{history}"
                            f"\n\n# Competing Vendor Context\n{context}")
        #print("======= objection_prompt =========")
        #print(objection_prompt)
        objection_response = self.model.invoke([
            SystemMessage(content=("You are a skeptical customer analyzing vendor responses. "
                                   f"You have been chatting with {vendor_to_object} about your inquery: {self.initial_prompt}.\n"
                                   f"please respond with the voice of the customer, as if you was asking {vendor_to_object} directly.")),
            HumanMessage(content=objection_prompt)
        ])
        return objection_response.content.strip()

    def _create_further_discovery_questions(self, vendor):
        """
        Generate further discovery questions for the vendor based on the conversation history.
        """
        history = "\n\n".join([msg.content for msg in self.messages[vendor]])
        question_prompt = f"Based on the following conversation history with {vendor}, generate additional discovery questions for them:\n{history}"
        #print("======= discovery_prompt =========")
        #print(question_prompt)
        question_response = self.model.invoke([
            SystemMessage(content=(f"You are a customer seeking more clarity in discussions with {vendor} about your inquiry: {self.initial_prompt}. "
                                   f"Please respond with the voice of the customer, as if the customer was asking the {vendor} directly. ")),
            HumanMessage(content=question_prompt)
        ])
        return question_response.content.strip()

    def create_questions(self, vendor, round):
        """
        Generate objections and discovery questions for a given vendor.
        """
        if round < 2:
            return self.initial_prompt
        objections = self._make_objections(vendor)
        discovery_questions = self._create_further_discovery_questions(vendor)
        return f"Objections:\n{objections}\n\nDiscovery Questions:\n{discovery_questions}"

    def log(self, vendor, my_questions, response):
        """
        Log the customer’s questions and the vendor’s response with a timestamp.
        """
        self.messages[vendor].append(HumanMessage(content=my_questions))
        self.messages[vendor].append(AIMessage(content=response))

        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"## Vendor {vendor}: {timestamp} \n\n**Questions:**\n{my_questions}\n\n Response:**\n{response}\n\n"

        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(log_entry)

    def _log_final_decision(self, reason='UNKNOWN'):
        """
        Read log and create a final decision based on the accumulated interactions.
        """
        with open(self.log_file, "r", encoding="utf-8") as file:
            log_content = file.read()

        decision_prompt = f"Based on the following customer-vendor interactions, generate a final decision including a clear choice and justifications:\n\n{log_content}"
        final_decision_response = self.model.invoke([
            SystemMessage(content="You are a customer making a final decision based on vendor interactions."),
            HumanMessage(content=decision_prompt)
        ])
        final_decision = final_decision_response.content.strip()

        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(f"# Final Decision\n## Reason: {reason}\n\n{final_decision}\n\n")

    def should_continue(self, round):
        """
        Determine whether the customer should continue based on the log or make a final decision.
        """
        with open(self.log_file, "r", encoding="utf-8") as file:
            log_content = file.read()
        if round > 3:
            self._log_final_decision('MAX_ITERATIONS')
            return False
        decision_prompt = (
            "Based on the current discussions, should we continue asking for more information "
            "or have we gathered enough to make a decision? Reply with 'continue' or 'stop'. "
            "You should do at least 2 rounds to get responses. Please stop no matter what after 4 rounds. "
            f"You are currently on round {round}.\n\n{log_content}"
        )

        decision_response = self.model.invoke([
            SystemMessage(content="You are a customer evaluating solutions."),
            HumanMessage(content=decision_prompt)
        ])
        decision = decision_response.content.strip().lower()

        if decision == "stop":
            self._log_final_decision('ENOUGH_INFO')
            return False
        return True


# Define the Product Marketer and Architect agent
class ProductMarketerArchitect:
    def __init__(self, product_name, initial_prompt):
        self.product_name = product_name
        self.messages = [SystemMessage(content=f"You are a product marketer for {product_name}.")]
        self.messages.append(HumanMessage(content=initial_prompt))
        self.model = ChatOpenAI(model_name="gpt-4o")
        self.query_creation_model = ChatOpenAI(model_name="gpt-4o").with_structured_output(Queries)
        self.sales_playbook_path = f"simulated-content/{product_name.replace(' ', '_')}_Sales_Playbook.md"
        self.changelog_path = f"simulated-content/{product_name.replace(' ', '_')}_Changelog.md"
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

    def read_sales_playbook(self):
        if os.path.exists(self.sales_playbook_path):
            with open(self.sales_playbook_path, "r", encoding="utf-8") as file:
                return file.read()
        return ""

    def web_search(self, query):
        search_results = self.search_tool.invoke({"query": query})
        res = json.dumps(search_results[:5], indent=2)  # Convert the first 5 results to a formatted JSON string
        return res

    def update_sales_playbook(self, customer_message, response):
        # Read current sales playbook
        current_playbook = self.read_sales_playbook()

        # Generate sales playbook updates
        update_prompt = (f"Based on the current sales playbook, the customer question, and your response, rewrite the "
                         f"sales playbook accordingly.\n\nCurrent Sales Playbook:\n{current_playbook}\n\n"
                         f"Customer Message: {customer_message}\n\nYour Response: {response}\n\n")
        update_response = self.model.invoke([
            SystemMessage(content="You are a strategic sales expert updating a sales playbook. The salesplaybook "
                                  "should include a summary of everything you need to\n"
                                  "- explain your unique differentiated value proposition\n"
                                  "- handle common customer objects\n"
                                  "- respond to common discovery questions\n"
                                  "Make sure to cite sources where possible so we can trace information"),
            HumanMessage(content=update_prompt)
        ])
        update_text = update_response.content

        # Generate a changelog entry
        changelog_prompt = (f"Summarize the changes made to the sales playbook based on the new customer "
                            f"conversation:\n\nLast Playbook:\n{current_playbook}\n\nCustomer Message: {customer_message}\n\nYour Response: {response}\n\nUpdated Playbook:\n{update_text}\n\n")
        changelog_response = self.model.invoke([
            SystemMessage(content="You are an expert summarizing changes to a sales playbook."),
            HumanMessage(content=changelog_prompt)
        ])
        timestamp = datetime.datetime.now().isoformat()
        changelog_text = f"## TimeStamp: {timestamp}\n{changelog_response.content}\n\n"

        # Write updates to the sales playbook
        with open(self.sales_playbook_path, "w", encoding="utf-8") as file:
            file.write(f"##{update_text}\n\n")

        # Write updates to the changelog
        with open(self.changelog_path, "a", encoding="utf-8") as file:
            file.write(f"Updated sales playbook based on new customer conversation.\n\n{changelog_text}\n\n")

    def converse_with_customer(self, customer_message):
        print("\t summarizing previous conversation history....")
        # Summarize previous messages
        conversation_summary_prompt = (
            "Summarize the key points and objections from the following conversation history "
            "in a concise manner for context preservation. Keep it under 300 tokens.\n\n"
            f"Conversation History:\n{self.messages}"
        )

        summary_response = self.model.invoke([
            SystemMessage(content="You are a helpful assistant summarizing a conversation."),
            HumanMessage(content=conversation_summary_prompt)
        ])
        summarized_history = summary_response.content.strip()

        print("\t conducting web searches....")
        # Read sales playbook
        sales_playbook_content = self.read_sales_playbook()
        # Develop search query
        search_query_prompt = (f"Based on the sales playbook content, develop the best search queries to gather "
                               f"missing or supporting information for {self.product_name}."
                               f"\n\nSales Playbook:\n{sales_playbook_content}\n\n. Limit to 5 queries max")
        search_queries = self.query_creation_model.invoke([
            SystemMessage(content="You are a marketing expert optimizing search queries."),
            HumanMessage(content=search_query_prompt)
        ])

        # Perform web search
        additional_info = ""
        for search_query in tqdm(search_queries.queries, desc="Fetching web search results", unit="query"):
            result = self.web_search(search_query)
            additional_info += f"\n\n## {search_query}\n{result}"

        print("\t formulating response....")
        # Combine information for response
        self.messages.append(HumanMessage(content=customer_message))
        response = self.model.invoke([
            HumanMessage(content=f"Customer Questions: \n{customer_message}"),
            HumanMessage(content=f"Here is a summary of your previous conversation so far:\n{summarized_history}"),
            HumanMessage(content=f"Here is your sales playbook: \n{sales_playbook_content}"),
            HumanMessage(content=f"Here is additional information from the web:\n{additional_info}")])
        self.messages.append(response)

        print("\t updating sales playbook....")
        self.update_sales_playbook(customer_message, response.content)

        return response.content


# Conversation simulation
with open("agent-info.json", "r", encoding="utf-8") as file:
    agent_info = json.load(file)

print('========= agent_info ====================')
print(agent_info)
customer_prompt = agent_info['customer']['initial_prompt']
customer = Customer(customer_prompt, [competitor['product_name'] for competitor in agent_info['competitors']])

competitors = [
    ProductMarketerArchitect(product_name=competitor['product_name'], initial_prompt=competitor['initial_prompt']) for
    competitor in agent_info['competitors']]

iteration = 1

while True:
    print(f"\n\n===================")
    print(f"Iteration: {iteration}")
    for competitor in competitors:
        print(f"\n--------------------")
        print(f"Competitor: {competitor.product_name}")
        print(f"customer creating question(s)...")
        questions = customer.create_questions(competitor.product_name, iteration)
        print(f"customer asking:\n{questions}")
        print(f"competitor responding & refining playbook...")
        response = competitor.converse_with_customer(questions)
        print(f"customer logging...")
        customer.log(competitor.product_name, questions, response)
    print(f"customer evaluating....")
    if not customer.should_continue(iteration):
        print(f"customer reached final decision")
        break
    iteration += 1
