from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import TypedDict

import os

# 1. Definir estado
class GraphState(TypedDict):
    prompt: str
    analysis: str
    improved_prompt: str

# 2. Configurar modelo y templates corregidos
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=os.environ["OPENAI_API_KEY"])

analyze_prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="""
    Genera SOLO un JSON válido con este formato:
    {{
        "well_written": boolean,
        "missing_content": boolean,
        "inappropriate_content": boolean,
        "needs_improvement": boolean,
        "suggestions": ["sugerencia 1", "sugerencia 2", "sugerencia 3"]
    }}
    Analiza este prompt: '{prompt}'
    """
)

improve_prompt_template = PromptTemplate(
    input_variables=["prompt", "suggestions"],
    template="""
    [INSTRUCCIONES]
    1. Mejora este prompt para que sea más efectivo: 
    **Prompt original**: {prompt}
    
    2. Aplica estas sugerencias: {suggestions}
    
    3. **Formato requerido**:
    - Mantener el objetivo original
    - Añadir detalles específicos (audiencia, estilo, elementos clave)
    - No generar el contenido final, solo el prompt mejorado
    
    Respuesta SOLO con el nuevo prompt optimizado.
    """
)

# 3. Nodos del grafo
def analyze_prompt(state: GraphState):
    chain = analyze_prompt_template | llm
    return {"analysis": chain.invoke({"prompt": state["prompt"]}).content}

def improve_prompt(state: GraphState):
    chain = improve_prompt_template | llm
    return {"improved_prompt": chain.invoke({
        "prompt": state["prompt"],
        "suggestions": state["analysis"]
    }).content}

# 4. Configurar grafo
workflow = StateGraph(GraphState)
workflow.add_node("analyze", analyze_prompt)
workflow.add_node("improve", improve_prompt)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "improve")
workflow.add_edge("improve", END)

app = workflow.compile()

# 5. Función de ejecución
def run_flow(prompt: str):
    result = app.invoke({"prompt": prompt})
    return result.get("improved_prompt", "Error en la generación")