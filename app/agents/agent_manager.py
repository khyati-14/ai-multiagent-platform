from crewai import Agent, Task, Crew
from app.services.rag_pipeline import query_rag

def create_agents():
    retriever_agent = Agent(
        role='Data Retriever',
        goal='Retrieve the most relevant context for user queries',
        backstory='Specialist in extracting precise data from document embeddings',
        allow_delegation=False,
        verbose=True
    )

    analyst_agent = Agent(
        role='Analyst',
        goal='Analyze retrieved data and produce clear, concise answers',
        backstory='Expert in summarizing and interpreting data',
        allow_delegation=False,
        verbose=True
    )

    return retriever_agent, analyst_agent

def run_multiagent_query(question):
    retriever, analyst = create_agents()

    retrieval_task = Task(
        description=f"Retrieve context for: {question}",
        agent=retriever,
        func=lambda: query_rag(question)
    )

    analysis_task = Task(
        description=f"Analyze and answer: {question}",
        agent=analyst
    )

    crew = Crew(agents=[retriever, analyst], tasks=[retrieval_task, analysis_task])
    results = crew.kickoff()
    return results
