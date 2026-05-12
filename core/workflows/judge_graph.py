from typing import TypedDict, Annotated, List, Union, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from core.llm_models.router import ModelRouter
from core.llm_models.base_provider import JudgeResult
from core.modules.base import Base as BaseModule
from utils.logger import get_logger

logger = get_logger("JudgeGraph")

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    input_prompt: str
    generation: str
    judge_result: JudgeResult
    iterations: int
    max_iterations: int
    api_keys: dict

def generate_node(state: GraphState):
    """
    Generates a response using the primary model.
    """
    logger.info(f"Generating response (Iteration {state['iterations'] + 1})")
    
    # Create a temporary module for generation
    class TempModule(BaseModule):
        prompt: str = state['input_prompt']
        # If we have a previous judge result, we could append it to the prompt here
    
    if state['judge_result'] and state['judge_result'].improvements:
        prompt_with_feedback = f"{state['input_prompt']}\n\n[Previous Feedback]: {state['judge_result'].improvements}"
    else:
        prompt_with_feedback = state['input_prompt']
    
    module = TempModule(prompt=prompt_with_feedback)
    router = ModelRouter(module, api_keys=state['api_keys'])
    response = router.model_response(module)
    
    return {
        "generation": response,
        "iterations": state['iterations'] + 1
    }

def judge_node(state: GraphState):
    """
    Evaluates the generation using the judge logic.
    """
    logger.info("Judging response...")
    
    # Use any model to judge, here we use the router's evaluate_response which defaults to a judge model
    # We need a dummy router just to call the evaluate_response method
    class DummyModule(BaseModule):
        model: str = "gemini-2.0-flash" # Standard judge model

    router = ModelRouter(DummyModule(), api_keys=state['api_keys'])
    judge_result = router.evaluate_response(state['input_prompt'], state['generation'])
    
    logger.info(f"Judge Score: {judge_result.score}/10")
    return {
        "judge_result": judge_result
    }

def should_continue(state: GraphState):
    """
    Determines whether to continue iterating or end.
    """
    if state['judge_result'].score >= 8:
        logger.info("Quality threshold met. Ending.")
        return "end"
    elif state['iterations'] >= state['max_iterations']:
        logger.info("Max iterations reached. Ending.")
        return "end"
    else:
        logger.info("Quality threshold not met. Retrying...")
        return "continue"

def create_judge_graph():
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("generate", generate_node)
    workflow.add_node("judge", judge_node)

    # Set Entry Point
    workflow.set_entry_point("generate")

    # Add Edges
    workflow.add_edge("generate", "judge")
    
    workflow.add_conditional_edges(
        "judge",
        should_continue,
        {
            "continue": "generate",
            "end": END
        }
    )

    return workflow.compile()

if __name__ == "__main__":
    # Example usage
    from utils.env_ops import get_keys_dict
    
    app = create_judge_graph()
    
    initial_state = {
        "input_prompt": "Explain quantum entanglement to a 5-year old.",
        "generation": "",
        "judge_result": None,
        "iterations": 0,
        "max_iterations": 3,
        "api_keys": get_keys_dict()
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    print("\n" + "="*50)
    print("FINAL GENERATION:")
    print(final_state['generation'])
    print("="*50)
    print(f"FINAL SCORE: {final_state['judge_result'].score}")
    print(f"ITERATIONS: {final_state['iterations']}")
