import streamlit as st
import openai
import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import uuid
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o"  # Using GPT-4o for advanced reasoning capabilities

# ============================================================================
# Memory and State Management
# ============================================================================

class Memory:
    """Memory system for the AI agent to store conversation history and context."""
    
    def __init__(self, max_short_term_memory: int = 10):
        """Initialize memory systems.
        
        Args:
            max_short_term_memory: Maximum number of interactions to keep in short-term memory
        """
        # Short-term memory for recent conversations
        self.short_term_memory: List[Dict[str, Any]] = []
        self.max_short_term_memory = max_short_term_memory
        
        # Long-term memory for important information and context
        self.long_term_memory: Dict[str, Any] = {
            "user_preferences": {},
            "important_facts": [],
            "completed_tasks": [],
            "session_stats": {
                "start_time": datetime.now().isoformat(),
                "interactions": 0,
                "successful_tasks": 0,
                "failed_tasks": 0
            }
        }
    
    def add_interaction(self, user_input: str, agent_response: str, 
                       reasoning: Optional[str] = None, 
                       task_success: Optional[bool] = None) -> None:
        """Add a new interaction to short-term memory.
        
        Args:
            user_input: The user's input message
            agent_response: The agent's response message
            reasoning: The agent's reasoning process (optional)
            task_success: Whether the task was successful (optional)
        """
        # Create new memory entry
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": agent_response,
            "reasoning": reasoning,
            "task_success": task_success
        }
        
        # Add to short-term memory
        self.short_term_memory.append(memory_entry)
        
        # Trim short-term memory if it exceeds maximum size
        if len(self.short_term_memory) > self.max_short_term_memory:
            self.short_term_memory.pop(0)
        
        # Update session statistics
        self.long_term_memory["session_stats"]["interactions"] += 1
        if task_success is not None:
            if task_success:
                self.long_term_memory["session_stats"]["successful_tasks"] += 1
            else:
                self.long_term_memory["session_stats"]["failed_tasks"] += 1
    
    def add_to_long_term_memory(self, category: str, information: Any) -> None:
        """Store important information in long-term memory.
        
        Args:
            category: Category of information (e.g., "user_preferences", "important_facts")
            information: The information to store
        """
        if category in self.long_term_memory:
            if isinstance(self.long_term_memory[category], list):
                self.long_term_memory[category].append(information)
            elif isinstance(self.long_term_memory[category], dict):
                self.long_term_memory[category].update(information)
        else:
            self.long_term_memory[category] = information
    
    def get_conversation_history(self, max_entries: Optional[int] = None) -> List[Dict[str, str]]:
        """Get recent conversation history formatted for OpenAI.
        
        Args:
            max_entries: Maximum number of entries to return (default: all in short-term memory)
        
        Returns:
            List of conversation messages in OpenAI format
        """
        history = []
        
        entries = self.short_term_memory
        if max_entries is not None:
            entries = entries[-max_entries:]
        
        for entry in entries:
            history.append({"role": "user", "content": entry["user_input"]})
            history.append({"role": "assistant", "content": entry["agent_response"]})
        
        return history
    
    def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search through memory for relevant information.
        
        Args:
            query: Search query
        
        Returns:
            List of relevant memory entries
        """
        # This is a simple implementation - a production version would use embeddings and vector search
        results = []
        
        # Search through short-term memory
        for entry in self.short_term_memory:
            if query.lower() in entry["user_input"].lower() or query.lower() in entry["agent_response"].lower():
                results.append(entry)
        
        # Search through long-term memory important facts
        for fact in self.long_term_memory["important_facts"]:
            if isinstance(fact, str) and query.lower() in fact.lower():
                results.append({"source": "important_facts", "content": fact})
        
        return results
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's memory state."""
        return {
            "short_term_memory_size": len(self.short_term_memory),
            "long_term_memory_categories": list(self.long_term_memory.keys()),
            "session_stats": self.long_term_memory["session_stats"]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        return {
            "short_term_memory": self.short_term_memory,
            "long_term_memory": self.long_term_memory,
            "max_short_term_memory": self.max_short_term_memory
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create Memory instance from dictionary."""
        memory = cls(max_short_term_memory=data.get("max_short_term_memory", 10))
        memory.short_term_memory = data.get("short_term_memory", [])
        memory.long_term_memory = data.get("long_term_memory", {})
        return memory


# ============================================================================
# Agent Tools and Capabilities
# ============================================================================

class Tool:
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        """Initialize a tool.
        
        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description
    
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool functionality.
        
        Returns:
            Dictionary with the execution result
        """
        raise NotImplementedError("Tool must implement execute method")
    
    def to_function_spec(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function specification format.
        
        Returns:
            Function specification dictionary
        """
        raise NotImplementedError("Tool must implement to_function_spec method")


class WebSearchTool(Tool):
    """Tool for searching the web for information."""
    
    def __init__(self):
        """Initialize web search tool."""
        super().__init__(
            name="web_search",
            description="Search the web for information on a specific topic. Do not use this for analyzing uploaded files."
        )
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute a web search.
        
        Args:
            query: Search query
        
        Returns:
            Dictionary with search results
        """
        # In a production app, use a real search API like Serper.dev, SerpAPI, or Bing API
        # This is a simplified mock implementation
        st.info(f"🔍 Searching the web for: {query}")
        time.sleep(1)  # Simulate API call delay
        
        # Check if this is about a file analysis, which would be inappropriate for web search
        if any(term in query.lower() for term in ['uploaded file', 'pdf file', 'csv file', 'excel file']):
            return {
                "status": "error",
                "query": query,
                "error": "Web search is not appropriate for analyzing uploaded files. Please use the analyze_data tool instead."
            }
        
        return {
            "status": "success",
            "query": query,
            "results": [
                {
                    "title": f"Result for {query}",
                    "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                    "snippet": f"This is a simulated search result for {query}. In a production environment, this would be a real result from a search API."
                }
            ]
        }
    
    def to_function_spec(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function specification format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }


class DataAnalysisTool(Tool):
    """Tool for analyzing data from CSV or other structured data."""
    
    def __init__(self):
        """Initialize data analysis tool."""
        super().__init__(
            name="analyze_data",
            description="Analyze data from uploaded files including CSV, Excel, Text, JSON, and PDF"
        )
    
    def execute(self, data_source: str, operation: str, 
               column: Optional[str] = None, 
               filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute data analysis operation.
        
        Args:
            data_source: Path to data file or name of cached DataFrame
            operation: Analysis operation (summary, filter, sort, etc.)
            column: Column name for operations that require it
            filters: Filter conditions
        
        Returns:
            Dictionary with analysis results
        """
        try:
            st.info(f"📊 Analyzing data: {operation} on {data_source}")
            
            # Check if we're working with uploaded files in the session state
            if hasattr(st, 'session_state') and 'uploaded_files' in st.session_state:
                # Try to find the file by name in uploaded_files
                file_found = False
                for file_id, file_info in st.session_state.uploaded_files.items():
                    if file_info['filename'] == data_source or data_source in file_info['filename'] or file_info['filename'] in data_source:
                        file_found = True
                        file_content = file_info['content']
                        file_type = file_info['type']
                        filename = file_info['filename']
                        
                        st.info(f"Found file: {file_info['filename']} ({file_info['size']} bytes)")
                        
                        # Handle PDF files
                        if filename.lower().endswith('.pdf'):
                            try:
                                import io
                                import PyPDF2
                                
                                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                                num_pages = len(pdf_reader.pages)
                                
                                text_content = ""
                                for page_num in range(num_pages):
                                    page = pdf_reader.pages[page_num]
                                    text_content += page.extract_text() + "\n\n"
                                
                                return {
                                    "status": "success",
                                    "operation": "pdf_analysis",
                                    "filename": filename,
                                    "num_pages": num_pages,
                                    "content_preview": text_content[:1000] + "..." if len(text_content) > 1000 else text_content,
                                    "full_text": text_content
                                }
                            except Exception as e:
                                return {
                                    "status": "error",
                                    "operation": "pdf_analysis",
                                    "error": f"Error analyzing PDF: {str(e)}"
                                }
                        
                        # Parse based on file type
                        if filename.lower().endswith('.csv') or 'csv' in file_type.lower():
                            # Parse CSV
                            import io
                            import pandas as pd
                            df = pd.read_csv(io.BytesIO(file_content))
                        elif filename.lower().endswith('.xlsx') or 'excel' in file_type.lower():
                            # Parse Excel
                            import io
                            import pandas as pd
                            df = pd.read_excel(io.BytesIO(file_content))
                        elif filename.lower().endswith('.txt'):
                            # For text files, just return the content
                            text_content = file_content.decode('utf-8', errors='replace')
                            return {
                                "status": "success",
                                "operation": "text_analysis",
                                "filename": filename,
                                "content_preview": text_content[:1000] + "..." if len(text_content) > 1000 else text_content,
                                "full_text": text_content
                            }
                        elif filename.lower().endswith('.json'):
                            # Parse JSON
                            import json
                            import pandas as pd
                            data = json.loads(file_content.decode('utf-8'))
                            if isinstance(data, list):
                                df = pd.DataFrame(data)
                            else:
                                # Handle nested JSON or single object
                                df = pd.json_normalize(data)
                        else:
                            return {
                                "status": "error",
                                "error": f"Unsupported file type: {file_info['type']} for file {file_info['filename']}"
                            }
                
                if not file_found:
                    # File not found, let the user know
                    available_files = [f_info['filename'] for f_id, f_info in st.session_state.uploaded_files.items()]
                    return {
                        "status": "error",
                        "error": f"File '{data_source}' not found in uploaded files. Available files: {', '.join(available_files)}"
                    }
            else:
                # No uploaded files, let the user know
                return {
                    "status": "error",
                    "error": "No files have been uploaded. Please upload a file first."
                }
            
            # Log basic dataframe info for tabular data
            st.info(f"DataFrame shape: {df.shape}")
            
            result = {"status": "success", "operation": operation, "filename": filename}
            
            if operation == "summary":
                result["data"] = df.describe().to_dict()
                result["columns"] = df.columns.tolist()
                result["shape"] = df.shape
                result["head"] = df.head(5).to_dict(orient="records")
                result["info"] = {
                    "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                    "non_null_counts": df.count().to_dict()
                }
            elif operation == "filter" and filters:
                # Apply filters (simplified)
                filtered_df = df
                for col, value in filters.items():
                    filtered_df = filtered_df[filtered_df[col] == value]
                result["data"] = filtered_df.to_dict(orient="records")
            elif operation == "sort" and column:
                result["data"] = df.sort_values(by=column).to_dict(orient="records")
            elif operation == "unique" and column:
                result["data"] = df[column].unique().tolist()
            elif operation == "correlation":
                # Only include numeric columns
                numeric_df = df.select_dtypes(include=['number'])
                if numeric_df.shape[1] > 1:
                    result["data"] = numeric_df.corr().to_dict()
                else:
                    result["data"] = {"warning": "Not enough numeric columns for correlation"}
            elif operation == "preview":
                result["head"] = df.head(10).to_dict(orient="records")
                result["columns"] = df.columns.tolist()
                result["shape"] = df.shape
            else:
                result["status"] = "error"
                result["error"] = "Invalid operation or missing parameters"
            
            return result
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def to_function_spec(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function specification format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_source": {
                            "type": "string",
                            "description": "The name of the file to analyze"
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["summary", "filter", "sort", "unique", "correlation", "preview", "pdf_analysis", "text_analysis"],
                            "description": "The analysis operation to perform"
                        },
                        "column": {
                            "type": "string",
                            "description": "Column name for operations that require it"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Filter conditions for filtering operations"
                        }
                    },
                    "required": ["data_source", "operation"]
                }
            }
        }


# ============================================================================
# Agentic AI Core
# ============================================================================

class AgentCore:
    """Core agent logic for autonomous reasoning, planning, and execution."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """Initialize agent core with tools and memory.
        
        Args:
            model: OpenAI model to use
        """
        self.model = model
        self.memory = Memory()
        self.tools = self._initialize_tools()
        
        # Agent settings
        self.settings = {
            "autonomy_level": 0.7,  # 0 to 1, how autonomous the agent is
            "verbose_reasoning": True,  # Whether to show detailed reasoning
            "self_reflection_enabled": True,  # Whether agent should reflect on actions
            "tool_usage_threshold": 0.6  # Threshold for determining when to use tools
        }
        
        # System message defines the agent's personality and capabilities
        self.system_message = """You are an autonomous AI agent with the ability to reason, plan, and execute tasks 
based on user input. Your goal is to be helpful, accurate, and efficient.

You have access to several tools to help you complete tasks. When appropriate, use these tools
to gather information or perform actions.

Always approach tasks by:
1. Understanding the user's request
2. Planning the steps needed to fulfill the request
3. Executing the necessary actions using your available tools
4. Reflecting on your actions and improving your approach

Present your thinking process clearly, and be transparent about your limitations.
You should balance being concise with being thorough."""
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize available tools for the agent.
        
        Returns:
            List of available tools
        """
        return [
            WebSearchTool(),
            DataAnalysisTool()
        ]
    
    def _get_function_specs(self) -> List[Dict[str, Any]]:
        """Get function specifications for all available tools.
        
        Returns:
            List of function specifications for OpenAI API
        """
        return [tool.to_function_spec() for tool in self.tools]
    
    def _get_tool_by_name(self, name: str) -> Optional[Tool]:
        """Get tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            Tool instance or None if not found
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate agent response.
        
        Args:
            user_input: User's input message
        
        Returns:
            Dictionary with agent response and metadata
        """
        start_time = time.time()
        
        # Check if we have an empty query after a file upload
        if user_input.strip() == "" and hasattr(st, 'session_state') and st.session_state.get("uploaded_files", {}):
            files_info = ", ".join([f_info["filename"] for f_id, f_info in st.session_state.uploaded_files.items()])
            return {
                "message": f"I see you've uploaded these files: {files_info}. What would you like me to do with them? I can analyze the data, extract insights, or answer specific questions about the content.",
                "success": True,
                "tools_used": []
            }
        
        # Check for file-related keywords in user input
        file_keywords = ["file", "pdf", "csv", "excel", "xlsx", "txt", "data", "analyze", "dataset", "insights", "extract"]
        has_file_reference = any(keyword in user_input.lower() for keyword in file_keywords)
        
        # If user is asking about files and files are uploaded, directly analyze them
        if has_file_reference and hasattr(st, 'session_state') and 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            # Just get the first file for now
            file_id = next(iter(st.session_state.uploaded_files))
            file_info = st.session_state.uploaded_files[file_id]
            filename = file_info["filename"]
            
            st.info(f"Directly analyzing file: {filename}")
            
            # Handle PDF files
            if filename.lower().endswith('.pdf'):
                try:
                    import io
                    import PyPDF2
                    
                    file_content = file_info['content']
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                    num_pages = len(pdf_reader.pages)
                    
                    text_content = ""
                    for page_num in range(min(5, num_pages)):  # Get first 5 pages or all if less
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text() + "\n\n"
                    
                    # Use OpenAI to analyze the content
                    analysis_prompt = f"""
                    Analyze the following content from the PDF file '{filename}':
                    
                    {text_content[:4000]}  # Truncate if very long
                    
                    Provide key insights, main topics, and a summary of this content.
                    """
                    
                    analysis_response = openai.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an insightful document analyst."},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        temperature=0.5,
                        max_tokens=1500
                    )
                    
                    analysis_result = analysis_response.choices[0].message.content
                    
                    return {
                        "message": f"I've analyzed the PDF file '{filename}' ({num_pages} pages). Here are the insights:\n\n{analysis_result}",
                        "success": True,
                        "processing_time": time.time() - start_time,
                        "tools_used": []
                    }
                except Exception as e:
                    return {
                        "message": f"I encountered an error while analyzing the PDF file: {str(e)}. Please try a different file or approach.",
                        "success": False,
                        "processing_time": time.time() - start_time,
                        "tools_used": []
                    }
            
            # Handle CSV or Excel files
            elif filename.lower().endswith(('.csv', '.xlsx')):
                try:
                    import io
                    import pandas as pd
                    
                    file_content = file_info['content']
                    
                    if filename.lower().endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(file_content))
                    else:  # Excel
                        df = pd.read_excel(io.BytesIO(file_content))
                    
                    # Basic stats
                    rows, cols = df.shape
                    columns = df.columns.tolist()
                    
                    # Sample data
                    sample = df.head(5).to_dict(orient="records")
                    
                    return {
                        "message": f"I've analyzed the data file '{filename}' which contains {rows} rows and {cols} columns.\n\n"
                                  f"The columns are: {', '.join(columns)}\n\n"
                                  f"Here's a sample of the data:\n{sample}\n\n"
                                  f"Would you like me to perform specific analysis on this data?",
                        "success": True,
                        "processing_time": time.time() - start_time,
                        "tools_used": []
                    }
                except Exception as e:
                    return {
                        "message": f"I encountered an error while analyzing the data file: {str(e)}. Please try a different file or approach.",
                        "success": False,
                        "processing_time": time.time() - start_time,
                        "tools_used": []
                    }
            
            # Handle text files
            elif filename.lower().endswith('.txt'):
                try:
                    file_content = file_info['content']
                    text_content = file_content.decode('utf-8', errors='replace')
                    
                    # Use OpenAI to analyze the content
                    analysis_prompt = f"""
                    Analyze the following content from the text file '{filename}':
                    
                    {text_content[:4000]}  # Truncate if very long
                    
                    Provide key insights, main topics, and a summary of this content.
                    """
                    
                    analysis_response = openai.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an insightful document analyst."},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        temperature=0.5,
                        max_tokens=1500
                    )
                    
                    analysis_result = analysis_response.choices[0].message.content
                    
                    return {
                        "message": f"I've analyzed the text file '{filename}'. Here are the insights:\n\n{analysis_result}",
                        "success": True,
                        "processing_time": time.time() - start_time,
                        "tools_used": []
                    }
                except Exception as e:
                    return {
                        "message": f"I encountered an error while analyzing the text file: {str(e)}. Please try a different file or approach.",
                        "success": False,
                        "processing_time": time.time() - start_time,
                        "tools_used": []
                    }
            
            # Other file types
            else:
                return {
                    "message": f"I see you want to analyze the file '{filename}', but I don't currently support this file type for direct analysis. I support PDF, CSV, Excel, and text files.",
                    "success": False,
                    "processing_time": time.time() - start_time,
                    "tools_used": []
                }
        
        # Handle file-related queries that are more specific or for non-uploaded files
        # Specific part about files, users might upload them next
        if has_file_reference and not (hasattr(st, 'session_state') and 'uploaded_files' in st.session_state and st.session_state.uploaded_files):
            # Regular parsing, but return more info about expected files
            task_analysis = self._analyze_task(user_input)
            return {
                "message": "It seems you're asking about a file, but I don't see any uploaded files. Please upload the file you'd like me to analyze using the 'Upload Files' section above.",
                "success": False,
                "processing_time": time.time() - start_time,
                "tools_used": []
            }
            
        # Handle all non-file requests with the regular pipeline
        # 1. Parse the user input to understand the task
        task_analysis = self._analyze_task(user_input)
        
        # Handle no-query case
        if task_analysis["task_type"] == "no_query":
            return {
                "message": "I'm not sure what you're asking. Could you please provide more details about what you'd like me to help you with?",
                "success": False,
                "tools_used": []
            }
        
        # 2. Plan the approach for handling the task
        plan = self._plan_approach(user_input, task_analysis)
        
        # 3. Execute the plan, using tools if needed
        execution_result = self._execute_plan(user_input, plan)
        
        # 4. Reflect on the execution and improve for next time
        reflection = self._reflect_on_execution(user_input, execution_result) if self.settings["self_reflection_enabled"] else None
        
        # 5. Prepare the final response
        response = self._prepare_response(user_input, execution_result, reflection)
        
        # 6. Save the interaction to memory
        self.memory.add_interaction(
            user_input=user_input,
            agent_response=response["message"],
            reasoning=response.get("reasoning"),
            task_success=response.get("success")
        )
        
        # Include additional metadata for the UI
        response["processing_time"] = time.time() - start_time
        response["tools_used"] = execution_result.get("tools_used", [])
        
        return response

    def _analyze_task(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to understand the task.
        
        Args:
            user_input: User's input message
        
        Returns:
            Task analysis results
        """
        # Check if this is just a file upload with no specific query
        if user_input.strip() == "":
            return {
                "task_type": "no_query",
                "entities": [],
                "complexity": "simple",
                "tools_recommended": [],
                "raw_analysis": "No specific query provided."
            }
            
        messages = [
            {"role": "system", "content": """You are a task analyzer. Your job is to analyze the user's request
and determine:
1. The type of task they're asking for
2. The key entities involved
3. Whether any tools would be helpful
4. The complexity level (simple, medium, complex)"""},
            {"role": "user", "content": f"Analyze this request: {user_input}"}
        ]
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,  # Low temperature for more deterministic analysis
            max_tokens=500
        )
        
        analysis_text = response.choices[0].message.content
        
        # Extract structured information from analysis
        # In a production system, you might use function calling or more sophisticated parsing
        analysis = {
            "task_type": "unknown",
            "entities": [],
            "complexity": "medium",
            "tools_recommended": [],
            "raw_analysis": analysis_text
        }
        
        # Simple parsing of the analysis
        if "search" in analysis_text.lower() or "look up" in analysis_text.lower():
            analysis["task_type"] = "information_retrieval"
            analysis["tools_recommended"].append("web_search")
        
        if "data" in analysis_text.lower() or "analyze" in analysis_text.lower() or "file" in analysis_text.lower():
            analysis["task_type"] = "data_analysis"
            analysis["tools_recommended"].append("analyze_data")
        
        if "simple" in analysis_text.lower():
            analysis["complexity"] = "simple"
        elif "complex" in analysis_text.lower():
            analysis["complexity"] = "complex"
        
        return analysis

    def _plan_approach(self, user_input: str, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the approach for handling the task.
        
        Args:
            user_input: User's input message
            task_analysis: Task analysis results
        
        Returns:
            Plan for handling the task
        """
        # Check for file-related keywords in user input
        file_keywords = ["file", "pdf", "csv", "excel", "xlsx", "txt", "data", "analyze", "dataset"]
        has_file_reference = any(keyword in user_input.lower() for keyword in file_keywords)
        
        uploaded_files_info = ""
        uploaded_files_exist = False
        
        # Check if files are uploaded
        if hasattr(st, 'session_state') and 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            uploaded_files_exist = True
            file_names = [file_info["filename"] for file_id, file_info in st.session_state.uploaded_files.items()]
            uploaded_files_info = f"The user has uploaded these files: {', '.join(file_names)}. "
        
        # If the user mentions files and files are uploaded, prioritize file analysis
        if has_file_reference and uploaded_files_exist:
            # Create a simple plan focused on file analysis
            return {
                "steps": [
                    "1. Identify the file(s) the user wants to analyze",
                    "2. Use the analyze_data tool to extract information from the file",
                    "3. Process and summarize the extracted information",
                    "4. Present insights to the user"
                ],
                "tools_to_use": ["analyze_data"],
                "raw_plan": f"The user wants to analyze uploaded files. {uploaded_files_info}I will use the analyze_data tool to extract and analyze content from the file(s), then present insights to the user."
            }
        
        # Normal planning for other types of queries
        messages = [
            {"role": "system", "content": """You are a planning agent. Based on the user's request and the task analysis,
create a step-by-step plan for completing the task. Consider:
1. What information is needed
2. What tools should be used
3. In what order actions should be performed
4. What potential challenges might arise

Be specific and practical in your plan."""},
            {"role": "user", "content": f"""
Request: {user_input}

Task Analysis:
{json.dumps(task_analysis, indent=2)}

{uploaded_files_info}

Create a step-by-step plan for addressing this request.
"""}
        ]
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        plan_text = response.choices[0].message.content
        
        # Extract steps from the plan
        # A more sophisticated implementation could parse the steps more precisely
        steps = []
        for line in plan_text.split("\n"):
            if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "Step", "- ")):
                steps.append(line.strip())
        
        # If the user is asking about files but we didn't automatically detect it
        if uploaded_files_exist and not task_analysis.get("tools_recommended"):
            task_analysis["tools_recommended"] = ["analyze_data"]
        
        return {
            "steps": steps,
            "tools_to_use": task_analysis.get("tools_recommended", []),
            "raw_plan": plan_text
        }

    def _execute_plan(self, user_input: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan, using tools if needed.
        
        Args:
            user_input: User's input message
            plan: Plan for handling the task
        
        Returns:
            Execution results
        """
        # Start with fresh messages, don't include past conversation
        messages = [
            {"role": "system", "content": f"""{self.system_message}"""},
            {"role": "user", "content": user_input}
        ]
        
        # Add information about uploaded files to the system message if they exist
        if hasattr(st, 'session_state') and 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            uploaded_files_info = "You have access to these files:\n"
            for file_id, file_info in st.session_state.uploaded_files.items():
                uploaded_files_info += f"- {file_info['filename']} ({file_info['size']} bytes, type: {file_info['type']})\n"
            
            # Update the system message with file information
            messages[0]["content"] += f"\n\n{uploaded_files_info}"
        
        # Add the plan to the system message
        plan_info = f"\n\nYour plan to solve this task is:\n{plan.get('raw_plan', 'No plan available')}"
        messages[0]["content"] += plan_info
        
        # Clean up messages to ensure no null content
        for i, msg in enumerate(messages):
            if msg.get("content") is None:
                messages[i]["content"] = ""  # Replace None with empty string
        
        # Prepare the list of used tools
        tools_used = []
        
        # Maximum number of tool calls to prevent infinite loops
        max_tool_calls = 5
        tool_calls_count = 0
        
        # Initialize response content
        final_response = None
        reasoning_steps = []
        
        # Function execution loop
        while tool_calls_count < max_tool_calls:
            try:
                # For debugging
                if st.session_state.get("debug_mode", False):
                    st.write("API Request Messages:")
                    for i, msg in enumerate(messages):
                        st.write(f"{i}: {msg['role']} - {msg.get('content', '')[:50]}...")
                
                # Call the OpenAI API with function specifications
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1500
                }
                
                # Only add tools and tool_choice if tools are available
                if plan.get("tools_to_use"):
                    api_params["tools"] = self._get_function_specs()
                    api_params["tool_choice"] = "auto"
                    
                response = openai.chat.completions.create(**api_params)
                
                assistant_message = response.choices[0].message
                
                # Add the assistant message to our conversation
                assistant_msg = {"role": "assistant", "content": assistant_message.content or ""}
                
                # If the message has tool calls, add them to the message
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    assistant_msg["tool_calls"] = assistant_message.tool_calls
                
                messages.append(assistant_msg)
                
                # Check if the model has chosen to use a tool
                if not hasattr(assistant_message, 'tool_calls') or not assistant_message.tool_calls:
                    # No tool calls, we have the final response
                    final_response = assistant_message.content or ""  # Use empty string instead of None
                    break
                
                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_calls_count += 1
                    
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Find the tool and execute it
                    tool = self._get_tool_by_name(function_name)
                    if tool:
                        # Record the tool usage
                        tools_used.append({
                            "name": function_name,
                            "arguments": function_args
                        })
                        
                        # Execute the tool
                        tool_result = tool.execute(**function_args)
                        
                        # Add the tool result to the messages - must immediately follow the assistant message with tool_calls
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(tool_result)
                        })
                        
                        # Record reasoning step
                        reasoning_steps.append({
                            "step": f"Tool execution: {function_name}",
                            "input": function_args,
                            "output": tool_result
                        })
                    else:
                        # Tool not found
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps({
                                "status": "error",
                                "error": f"Tool {function_name} not found"
                            })
                        })
            except Exception as e:
                st.error(f"Error in execute_plan: {str(e)}")
                
                # For more serious debugging
                if st.session_state.get("debug_mode", False):
                    st.write("Last messages:")
                    for i in range(min(5, len(messages))):
                        if i >= len(messages) - 5:
                            st.write(f"{len(messages)-i}: {messages[i]['role']} - {messages[i].get('content', '')[:50]}...")
                
                return {
                    "message": f"I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question.",
                    "tools_used": tools_used,
                    "reasoning_steps": reasoning_steps,
                    "success": False
                }
        
        # If we've exhausted tool calls without a final response, generate one
        if final_response is None:
            # Ask for a final response based on the tool results
            messages.append({
                "role": "user",
                "content": "Based on the information gathered, please provide your final response to my request."
            })
            
            try:
                final_message_response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1500
                )
                
                final_response = final_message_response.choices[0].message.content or ""  # Use empty string instead of None
            except Exception as e:
                st.error(f"Error getting final response: {str(e)}")
                final_response = "I was unable to generate a final response due to an error. Please try again or simplify your request."
        
        return {
            "message": final_response,
            "tools_used": tools_used,
            "reasoning_steps": reasoning_steps,
            "success": len(tools_used) > 0 or final_response is not None
        }

    def _reflect_on_execution(self, user_input: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on the execution and generate insights for improvement.
        
        Args:
            user_input: User's input message
            execution_result: Execution results
        
        Returns:
            Reflection results
        """
        messages = [
            {"role": "system", "content": """You are a reflective agent. Your job is to analyze how the task was handled
and identify areas for improvement. Consider:
1. Were the right tools used?
2. Was the approach efficient?
3. Was the response clear and helpful?
4. What could be done better next time?

Be constructive and specific in your feedback."""},
            {"role": "user", "content": f"""
User Request: {user_input}

Tools Used: {json.dumps(execution_result.get('tools_used', []), indent=2)}

Final Response: {execution_result.get('message', 'No response generated')}

Please analyze this execution and provide insights for improvement.
"""}
        ]
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_tokens=500
        )
        
        reflection_text = response.choices[0].message.content
        
        # Store insights in long-term memory for future improvement
        self.memory.add_to_long_term_memory("execution_insights", {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "reflection": reflection_text
        })
        
        return {
            "reflection": reflection_text,
            "areas_for_improvement": [line.strip() for line in reflection_text.split("\n") if line.strip().startswith(("- ", "• "))]
        }

    def _prepare_response(self, user_input: str, execution_result: Dict[str, Any], 
                         reflection: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare the final response to the user.
        
        Args:
            user_input: User's input message
            execution_result: Execution results
            reflection: Reflection results (optional)
        
        Returns:
            Final response dictionary
        """
        # Start with the message from execution
        response = {
            "message": execution_result.get("message", "I couldn't generate a response."),
            "success": execution_result.get("success", False)
        }
        
        # Include reasoning if verbose mode is enabled
        if self.settings["verbose_reasoning"]:
            reasoning_content = []
            
            # Add reasoning steps
            for i, step in enumerate(execution_result.get("reasoning_steps", []), 1):
                reasoning_content.append(f"{i}. {step['step']}")
            
            # Add reflection insights if available
            if reflection and reflection.get("areas_for_improvement"):
                reasoning_content.append("\nReflection:")
                for insight in reflection.get("areas_for_improvement"):
                    reasoning_content.append(f"- {insight}")
            
            response["reasoning"] = "\n".join(reasoning_content)
        
        return response
    
    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Update agent settings.
        
        Args:
            new_settings: New settings to apply
        """
        self.settings.update(new_settings)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent.
        
        Returns:
            Dictionary with agent state
        """
        return {
            "model": self.model,
            "memory": self.memory.to_dict(),
            "settings": self.settings,
            "tools": [tool.name for tool in self.tools]
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load agent state from a dictionary.
        
        Args:
            state: Agent state dictionary
        """
        self.model = state.get("model", self.model)
        self.memory = Memory.from_dict(state.get("memory", {}))
        self.settings.update(state.get("settings", {}))


# ============================================================================
# Streamlit UI Components
# ============================================================================

class StreamlitUI:
    """Streamlit UI components for the agentic AI application."""
    
    def __init__(self, agent_core: AgentCore):
        """Initialize UI with agent core.
        
        Args:
            agent_core: Agent core instance
        """
        self.agent_core = agent_core
        
        # Initialize session state if needed
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "feedback" not in st.session_state:
            st.session_state.feedback = {}
        
        if "file_upload_key" not in st.session_state:
            st.session_state.file_upload_key = 0
        
        if "settings_expanded" not in st.session_state:
            st.session_state.settings_expanded = False
        
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}
    
    def render_header(self) -> None:
        """Render the application header."""
        st.title("🤖 AgentFlow")
        
        with st.expander("ℹ️ About this application", expanded=False):
            st.markdown("""
            AgentFlow is an autonomous AI agent capable of reasoning, planning, and executing
            tasks based on your input. It can:
            
            - Research information and provide detailed answers
            - Analyze data from uploaded files
            - Remember your preferences and previous interactions
            - Break down complex tasks into manageable steps
            
            Try asking a question or providing a task for the assistant to complete!
            """)
            
        if st.session_state.get("debug_mode", False):
            st.warning("🛠️ Debug Mode Enabled")
            
            if hasattr(st, 'session_state') and 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
                st.write("### Debug: Uploaded Files")
                for file_id, file_info in st.session_state.uploaded_files.items():
                    st.write(f"- File ID: {file_id}")
                    st.write(f"  - Name: {file_info['filename']}")
                    st.write(f"  - Size: {file_info['size']} bytes")
                    st.write(f"  - Type: {file_info['type']}")
                    
                    # Display a small preview of file content if it's text-based
                    if 'content' in file_info and file_info['size'] > 0:
                        try:
                            content_preview = file_info['content'][:200]
                            if isinstance(content_preview, bytes):
                                content_preview = content_preview.decode('utf-8', errors='replace')
                            st.code(content_preview + "...", language="text")
                        except Exception as e:
                            st.error(f"Error previewing file content: {str(e)}")
    
    def render_sidebar(self) -> None:
        """Render the sidebar with settings and controls."""
        st.sidebar.title("Agent Controls")
        
        # Model selection
        model = st.sidebar.selectbox(
            "OpenAI Model",
            options=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        
        if model != self.agent_core.model:
            self.agent_core.model = model
        
        # Agent settings
        st.sidebar.subheader("Agent Settings")
        
        new_settings = {}
        new_settings["autonomy_level"] = st.sidebar.slider(
            "Autonomy Level",
            min_value=0.0,
            max_value=1.0,
            value=self.agent_core.settings["autonomy_level"],
            step=0.1,
            help="How autonomous the agent should be (0 = minimal, 1 = maximal)"
        )
        
        new_settings["verbose_reasoning"] = st.sidebar.checkbox(
            "Show Reasoning Process",
            value=self.agent_core.settings["verbose_reasoning"],
            help="Display the agent's reasoning process in responses"
        )
        
        new_settings["self_reflection_enabled"] = st.sidebar.checkbox(
            "Enable Self-Reflection",
            value=self.agent_core.settings["self_reflection_enabled"],
            help="Allow the agent to reflect on its performance and improve"
        )
        
        # Debug mode
        st.sidebar.checkbox(
            "Debug Mode", 
            value=st.session_state.get("debug_mode", False),
            key="debug_mode",
            help="Show detailed debugging information"
        )
        
        # Update settings if changed
        if any(new_settings[key] != self.agent_core.settings[key] for key in new_settings):
            self.agent_core.update_settings(new_settings)
        
        # Memory management
        st.sidebar.subheader("Memory Management")
        if st.sidebar.button("Clear Conversation History"):
            st.session_state.messages = []
            self.agent_core.memory = Memory()
            st.toast("Conversation history cleared!", icon="🗑️")
        
        # Memory stats
        memory_summary = self.agent_core.memory.get_memory_summary()
        
        with st.sidebar.expander("Memory Statistics", expanded=False):
            st.write(f"Short-term memory size: {memory_summary['short_term_memory_size']}")
            st.write(f"Total interactions: {memory_summary['session_stats']['interactions']}")
            st.write(f"Successful tasks: {memory_summary['session_stats']['successful_tasks']}")
            st.write(f"Failed tasks: {memory_summary['session_stats']['failed_tasks']}")
    
    def render_file_upload(self) -> None:
        """Render file upload section."""
        with st.expander("📁 Upload Files", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload a file for the agent to analyze",
                    type=["csv", "xlsx", "txt", "json", "pdf"],
                    key=f"file_uploader_{st.session_state.file_upload_key}"
                )
            
            with col2:
                st.write("")
                st.write("")
                if st.button("Clear Files"):
                    st.session_state.uploaded_files = {}
                    st.session_state.file_upload_key += 1
                    st.toast("Uploaded files cleared!", icon="🗑️")
            
            if uploaded_file is not None:
                try:
                    # Save the file to memory
                    file_content = uploaded_file.read()
                    file_id = hashlib.md5(file_content).hexdigest()
                    
                    # Store in session state with more debugging info
                    if file_id not in st.session_state.uploaded_files:
                        file_info = {
                            "filename": uploaded_file.name,
                            "content": file_content,
                            "size": len(file_content),
                            "type": uploaded_file.type,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.uploaded_files[file_id] = file_info
                        
                        # Make sure the file content isn't empty
                        if len(file_content) == 0:
                            st.warning(f"Warning: The file '{uploaded_file.name}' appears to be empty.")
                        else:
                            # Display success message
                            st.success(f"File '{uploaded_file.name}' uploaded successfully! ({len(file_content)} bytes)")
                        
                        # Reset file uploader
                        st.session_state.file_upload_key += 1
                        
                        # Add file info to the agent's memory
                        self.agent_core.memory.add_to_long_term_memory("uploaded_files", {
                            "filename": uploaded_file.name,
                            "size": len(file_content),
                            "type": uploaded_file.type,
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as e:
                    st.error(f"Error uploading file: {str(e)}")
            
            # Display uploaded files with more detailed info
            if st.session_state.uploaded_files:
                st.write("Uploaded Files:")
                for file_id, file_info in st.session_state.uploaded_files.items():
                    file_size_kb = file_info['size'] / 1024
                    st.text(f"📄 {file_info['filename']} ({file_size_kb:.2f} KB)")
                
                # Debug information
                if st.checkbox("Show debug info"):
                    st.write("Session state contains these files:")
                    for file_id, file_info in st.session_state.uploaded_files.items():
                        st.write(f"File ID: {file_id}")
                        st.write(f"Filename: {file_info['filename']}")
                        st.write(f"Size: {file_info['size']} bytes")
                        st.write(f"Type: {file_info['type']}")
                        st.write(f"Upload time: {file_info['timestamp']}")
                        st.write("---")
                
                st.info("To analyze these files, ask a specific question in the chat. For example: 'What insights can you extract from the uploaded file?' or 'Summarize the content of the CSV file.'")


    
    def render_chat_message(self, message: Dict[str, Any]) -> None:
        """Render a chat message.
        
        Args:
            message: Message dictionary
        """
        role = message.get("role", "")
        content = message.get("content", "")
        
        with st.chat_message(role):
            st.markdown(content)
            
            # Show reasoning if available
            if role == "assistant" and message.get("reasoning") and self.agent_core.settings["verbose_reasoning"]:
                with st.expander("View reasoning process", expanded=False):
                    st.markdown(message["reasoning"])
            
            # Show feedback buttons for assistant messages
            if role == "assistant" and "message_id" in message:
                message_id = message["message_id"]
                col1, col2 = st.columns([1, 8])
                
                with col1:
                    # Thumbs up button
                    thumbs_up = st.button(
                        "👍", 
                        key=f"thumbs_up_{message_id}",
                        help="This response was helpful"
                    )
                    
                    # Thumbs down button
                    thumbs_down = st.button(
                        "👎", 
                        key=f"thumbs_down_{message_id}",
                        help="This response needs improvement"
                    )
                
                # Handle feedback
                if thumbs_up:
                    st.session_state.feedback[message_id] = "positive"
                    st.toast("Thanks for your positive feedback!", icon="👍")
                
                elif thumbs_down:
                    st.session_state.feedback[message_id] = "negative"
                    
                    # Get improvement feedback
                    feedback_text = st.text_area(
                        "How could this response be improved?",
                        key=f"feedback_text_{message_id}"
                    )
                    
                    if st.button("Submit Feedback", key=f"submit_feedback_{message_id}"):
                        st.session_state.feedback[message_id] = {
                            "rating": "negative",
                            "feedback": feedback_text
                        }
                        st.toast("Thanks for your feedback!", icon="🙏")
    
    def render_chat_interface(self) -> None:
        """Render the chat interface."""
        # Display chat history
        for message in st.session_state.messages:
            self.render_chat_message(message)
        
        # Chat input
        if prompt := st.chat_input("What can I help you with today?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_placeholder = st.empty()
                    response = self.agent_core.process_user_input(prompt)
                    
                    # Add unique ID for feedback
                    message_id = str(uuid.uuid4())
                    
                    # Display response
                    response_placeholder.markdown(response["message"])
                    
                    # Show reasoning if available
                    if response.get("reasoning") and self.agent_core.settings["verbose_reasoning"]:
                        with st.expander("View reasoning process", expanded=False):
                            st.markdown(response["reasoning"])
                    
                    # Add tools used if any
                    if response.get("tools_used"):
                        with st.expander("Tools used", expanded=False):
                            for tool in response["tools_used"]:
                                st.write(f"- {tool['name']}")
                    
                    # Show processing time
                    st.caption(f"Processing time: {response.get('processing_time', 0):.2f} seconds")
                    
                    # Add feedback buttons
                    col1, col2 = st.columns([1, 8])
                    
                    with col1:
                        # Thumbs up button
                        thumbs_up = st.button(
                            "👍", 
                            key=f"thumbs_up_{message_id}",
                            help="This response was helpful"
                        )
                        
                        # Thumbs down button
                        thumbs_down = st.button(
                            "👎", 
                            key=f"thumbs_down_{message_id}",
                            help="This response needs improvement"
                        )
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["message"],
                "reasoning": response.get("reasoning"),
                "tools_used": response.get("tools_used", []),
                "message_id": message_id
            })
    
    def render(self) -> None:
        """Render the complete UI."""
        self.render_header()
        self.render_sidebar()
        self.render_file_upload()
        self.render_chat_interface()


# ============================================================================
# Application Entry Point
# ============================================================================

def main():
    """Main application entry point."""
    # Set page configuration
    st.set_page_config(
        page_title="AgentFlow",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f3ff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Debug mode toggle in sidebar
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Initialize agent if not already in session state
    if "agent" not in st.session_state:
        # Check for API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            if not openai_api_key:
                st.warning("Please enter your OpenAI API key in the sidebar to continue.")
                st.stop()
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize agent
        st.session_state.agent = AgentCore()
    
    # Create and render UI
    ui = StreamlitUI(st.session_state.agent)
    ui.render()


if __name__ == "__main__":
    main()