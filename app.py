# app.py

from flask import Flask, render_template, request
from langgraph.graph import Graph
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
import openai
import os
import requests
import json 
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Configurar API Keys

api_key = os.getenv("OPENAI_API_KEY")
AZURE_KEY = os.getenv("AZURE_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

# Configurar modelos
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Función para Azure Content Safety
def azure_safety_check(prompt: str) -> dict:
    """Realiza el chequeo de seguridad con Azure"""
    url = f"{AZURE_ENDPOINT}/contentsafety/text:shieldPrompt?api-version=2024-09-01"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": AZURE_KEY
    }
    data = {
        "userPrompt": prompt,
        "documents": [prompt]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

# Templates de análisis
analyze_prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="""Analiza el prompt y devuelve un JSON **válido** (usa comillas dobles) con:
{
    "well_written": boolean,
    "missing_content": boolean,
    "inappropriate_content": boolean,
    "needs_improvement": boolean,
    "suggestions": string
}

Prompt: {prompt}"""
)

improve_prompt_template = PromptTemplate(
    input_variables=["prompt", "suggestions"],
    template="""Mejora este prompt basado en las sugerencias:
    Sugerencias: {suggestions}
    Prompt original: {prompt}
    Devuelve solo el prompt mejorado."""
)

# Nodos del flujo de trabajo
def analyze_prompt(state):
    prompt = state["prompt"]
    analysis = RunnablePassthrough() | analyze_prompt_template | llm
    return {"prompt": prompt, "analysis": analysis.invoke({"prompt": prompt}).content}

def improve_prompt(state):
    suggestions = state["analysis"]
    improve_chain = RunnablePassthrough() | improve_prompt_template | llm
    return {**state, "improved_prompt": improve_chain.invoke({
        "prompt": state["prompt"],
        "suggestions": suggestions
    }).content}

def security_check(state):
    azure_result = azure_safety_check(state["prompt"])
    return {**state, "azure_check": azure_result}

# Configurar el grafo
workflow = Graph()
workflow.add_node("analyze", analyze_prompt)
workflow.add_node("improve", improve_prompt)
workflow.add_node("security", security_check)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "improve")
workflow.add_edge("improve", "security")

app_workflow = workflow.compile()

# Rutas de Flask
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        result = app_workflow.invoke({"prompt": prompt})
        
        # Procesar resultados
        azure_result = result.get("azure_check", {})
        #analysis   # Convertir string JSON a dict
        analysis_str = result.get("analysis", "{}")
        try: 
            analysis = json.loads(analysis_str)
        except json.JSONDecodeError:
            analysis = {"error": "Formato JSON inválido"}
        
        return render_template('result.html', 
                             prompt=prompt,
                             analysis=analysis,
                             improved=result.get("improved_prompt"),
                             azure=azure_result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 
    #co