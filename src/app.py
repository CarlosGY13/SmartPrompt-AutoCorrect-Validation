from agent import run_flow
import gradio as gr

# Interfaz con salida de solo lectura y debug integrado
def generate(input_text):
    try:
        print(f"\n🔍 Input recibido: {repr(input_text)}")  # Debug 1
        
        if not input_text.strip():
            print("⚠️ Error: Prompt vacío")
            return "[ERROR] Por favor ingresa un prompt válido"
            
        improved = run_flow(input_text)
        print(f"✅ Output generado:\n{improved}")  # Debug 2
        
        return improved
        
    except Exception as e:
        error_msg = f"🚨 Error en el procesamiento: {str(e)}"
        print(error_msg)  # Debug 3
        return error_msg

# Configuración de la interfaz
demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(
        label="Prompt Original",
        placeholder="Escribe tu prompt aquí...",
        lines=3
    ),
    outputs=gr.Textbox(
        label="Prompt Mejorado",
        interactive=False,  # Texto de solo lectura
        lines=5,
        show_copy_button=True  # Botón de copia automático (Gradio >=3.39)
    ),
    title="Optimizador de Prompts",
    allow_flagging="never"
)

# Ejecución con verificación de puerto
if __name__ == "__main__":
    print("Iniciando servidor... Verifica los mensajes debajo 👇")
    demo.launch(
        server_port=7860,
        show_error=True,
        debug=True  # Muestra errores detallados
    )