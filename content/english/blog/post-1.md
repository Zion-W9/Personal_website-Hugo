---
title: "Introduction to Langchain"
meta_title: ""
description: "A brief intro of Langchain"
date: 2023-08-02T05:00:00Z
image: "/images/Blog/Langchain.png"
categories: ["NLP", "LLM"]
author: "Zeen"
tags: ["Langchain", "Deep Lake"]
draft: false
---

## [LangChain](https://python.langchain.com/docs/get_started)

### Introducing LangChain

**LangChain**: A cutting-edge open-source framework designed to amplify the capabilities of your LLM (Language Learning Model).

### Core Features of LangChain

- **Prompt Management**: Efficiently manage and optimize prompts tailored for diverse tasks.
- **Task Subdivision**: Seamlessly transition between various subtasks within a primary task.
- **Data Augmentation Generation**: This involves a unique chaining process that collaborates with external data sources to fetch data for the generation phase. For instance, generating summaries from extensive texts or Q&A sessions based on specific data sources.
- **Agent Operations**: Agents execute distinct actions based on varying instructions until the entire operation concludes.
- **Evaluation Mechanism**: Traditional metrics often fall short when evaluating generative models. LangChain introduces an innovative approach by utilizing the language model itself for evaluations, offering supportive hints and chains.
- **Memory Management**: Efficiently handle intermediate states throughout the process.

To encapsulate, LangChain can be visualized as a system that manages and refines prompts throughout a process's lifecycle. It employs various agents to execute actions based on these prompts, utilizes memory to oversee intermediate states, and then harnesses chains to interlink different agents, forming a cohesive loop.

### Value Propositions of LangChain

- **Components**: These are abstract constructs designed for language model processing, accompanied by a suite of implementations for each concept. Regardless of your engagement with other LangChain features, these components are modular and user-friendly.
- **Pre-configured Chains**: These are structured amalgamations of components aimed at executing specific high-tier tasks. While these chains offer a quick start, the components ensure easy customization for intricate applications.

### Key Components of LangChain

- **Model I/O**: Interface for the language model.
- **Data Connection**: Interface tailored for specific tasks.
- **Chains**: Construct a series of calls.
- **Agents**: Upon receiving high-level directives, the chain decides the appropriate tools.
- **Memory**: Retain application state between chain runs.
- **Callbacks**: Document and relay any intermediate chain steps.
- **Indexes**: Techniques to structure documents for optimal LLM interaction.

### Data Connection: A Deep Dive

LLM applications often necessitate user-specific data that isn't part of the model's foundational training set. LangChain offers a comprehensive toolkit for data handling:

- **Document Loader**: Fetch documents from a plethora of sources.
- **Document Converter**: Segment documents, eliminate superfluous ones, and more.
- **Text Embedding Model**: Convert unstructured text into float lists.
- **Vector Storage**: Archive and search embedded data.
- **Retriever**: Efficiently search through your data.

### Visual Representation of Data Connection

![Data Connection Flow](https://python.langchain.com/assets/images/data_connection-c42d68c3d092b85f50d08d4cc171fc25.jpg)

### Setting Up Data Connection: Document Loader

Python installation package command:

```bash
pip install langchain
pip install unstructured
pip install jq
```

**CSV basic usage**

```python3
import os
from pathlib import Path

from langchain.document_loaders import UnstructuredCSVLoader _
from langchain.document_loaders.csv_loader import CSVLoader
EXAMPLE_DIRECTORY = file_path = Path(__file__ ) .parent .parent / "examples"


def test_unstructured_csv_loader ( ) - > None:
"""Test unstructured loader."""
    file_path = os.path.join ( EXAMPLE_DIRECTORY , "stanley-cups.csv")
loader = UnstructuredCSVLoader (str( file_path ))
docs = loader.load ()
print(docs)
assert len (docs) == 1

def test_csv_loader ( ) :
  file_path = os.path.join ( EXAMPLE_DIRECTORY , "stanley-cups.csv")
loader = CSVLoader ( file_path )
docs = loader.load ()
print(docs)

test_unstructured_csv_loader ( ) _
test_csv_loader ( ) _
```

**File directory usage**

```python3
from langchain.document_loaders import DirectoryLoader , TextLoader _

text_loader_kwargs = { ' autodetect_encoding ': True}
loader = DirectoryLoader ( '../examples/',
glob="**/*.txt", # traverse txt files
              show_progress =True, # show progress
              use_multithreading =True, # use multithreading
              loader_cls = TextLoader , # use the way to load data
              silent_errors =True, #Continue when encountering an error
              loader_kwargs = text_loader_kwargs ) # You can use a dictionary to pass in parameters

docs = loader.load ()
print("\n")
print( docs[ 0])
```

**HTML usage**

```python3
from langchain.document_loaders import UnstructuredHTMLLoader , BSHTMLLoader _
loader = UnstructuredHTMLLoader ("../examples/example.html")
docs = loader.load ()
print( docs[ 0])

loader = BSHTMLLoader ("../examples/example.html")
docs = loader.load ()
print( docs[ 0])
```

**JSON Usage**

````python3
import json
from pathlib import Path
from pprint import pprint


file_path = '../examples/facebook_chat.json '
data = json. loads (Path( file_path ). read_text ())
pprint (data)

"""
{'image': {' creation_timestamp ': 1675549016, ' uri ': 'image_of_the_chat.jpg'},
' is_still_participant ': True ,
' joinable_mode ': {'link': '', 'mode': 1} ,
' magic_words ': [] ,
'messages': [{'content': 'Bye!',
' sender_name ': 'User 2' ,
' timestamp _ms ': 1675597571851},
{'content': 'Oh no worries! Bye',
' sender_name ': 'User 1' ,
' timestamp _ms ': 1675597435669},
{'content': 'No Im sorry it was my mistake, the blue one is not '
' for sale',
' sender_name ': 'User 2' ,
' timestamp _ms ': 1675596277579},
{'content': 'I thought you were selling the blue one!',
' sender_name ': 'User 1' ,
' timestamp _ms ': 1675595140251},
{'content': ' Im not interested in this bag. Im interested in the '
' blue one!',
' sender_name ': 'User 1' ,
' timestamp _ms ': 1675595109305},
{'content': 'Here is $129',
' sender_name ': 'User 2' ,
' timestamp _ms ': 1675595068468},
{'photos': [{' creation_timestamp ': 1675595059,
' uri ': 'url_of_some_picture.jpg'}],
' sender_name ': 'User 2' ,
' timestamp _ms ': 1675595060730},
{'content': 'Online is at least $100',
' sender_name ': 'User 2' ,
' timestamp _ms ': 1675595045152},
{'content': 'How much do you want?',
' sender_name ': 'User 1' ,
' timestamp _ms ': 1675594799696},
{'content': ' Goodmorning ! $50 is too low.',
' sender_name ': 'User 2' ,
' timestamp _ms ': 1675577876645},
{'content': 'Hi! Im interested in your bag. Im offering $50. Let '
' I know if you are interested. Thanks!',
' sender_name ': 'User 1' ,
' timestamp _ms ': 1675549022673}],
'participants': [{'name': 'User 1'}, {'name': 'User 2'}],
' thread _path ': 'inbox/User 1 and User 2 chat',
'title': 'User 1 and User 2 chat'}
"""
Load data using langchain :
```python
from langchain.document_loaders import JSONLoader _
loader = JSONLoader (
    file_path ='../examples/ facebook_chat.json ',
    jq_schema ='.messages[].content' # will report an error Expected page_content is string, got <class ' NoneType '> instead.
    page_content =False, # add this line after error reporting)

data = loader.load ()
print( data[ 0])
````

**PDF Usage**

```text
'''
first usage
'''
from langchain.document_loaders import PyPDFLoader _

loader = PyPDFLoader ("../examples/layout-parser-paper.pdf")
pages = loader.load_and_split ( )

print( pages[ 0])

'''
second usage
'''
from langchain.document_loaders import MathpixPDFLoader _

loader = MathpixPDFLoader (" example_data /layout-parser-paper.pdf")

data = loader.load ()
print( data[ 0])

'''
third usage
'''
from langchain.document_loaders import UnstructuredPDFLoader _

loader = UnstructuredPDFLoader ("../examples/layout-parser-paper.pdf")

data = loader.load ()
print( data[ 0])
```

### data connection - document conversion

Once files are loaded, adapting them to suit specific applications becomes paramount. A classic scenario involves segmenting a lengthy document into smaller fragments to align with your model's context window. LangChain is equipped with an array of integrated document conversion utilities. These tools facilitate effortless operations such as splitting, merging, filtering, and various other document manipulations.

**Text segmentation by character**

```text
state_of_the_union = """
Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They 
were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such 
nonsense. Mr Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, 
although he did have a very large moustache. Mrs Dursley was thin and blonde and had nearly twice the usual amount of neck, 
which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbours. The Dursleys had a 
small son called Dudley and in their opinion there was no finer boy anywhere. 
"""

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter (        
separator = "\n\n",
    chunk_size = 128, # Chunk length
    chunk_overlap = 10, # Overlapped text length
    length_function = len ,
)

texts = text_ splitter. create _documents ([ state_of_the_union ])
print( texts[ 0])

# Here metadatas are used to distinguish different documents
metadatas = [{"document": 1}, {"document": 2}]
documents = text_ splitter. create _documents ([ state_of_the_union , state_of_the_union ], metadatas = metadatas )
pprint (documents)

# Get the cut text
print( text_ splitter. split _text ( state_of_the_union )[0])
```

**Split the code**

```python3
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter ,
Language,
)

print([ e.value for e in Language]) # supported languages
print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)) # separator

PYTHON_CODE = """
def hello_world ( ) :
    print( "Hello, World!")

# Call the function
hello_world ( ) _
"""
python_splitter = RecursiveCharacterTextSplitter.from_language ( _
language= Language.PYTHON , chunk_size =50, chunk_overlap =0
)
python_docs = python_ splitter. create _documents ([PYTHON_CODE])
python_docs

"""
[ Document( page_content ='def hello_world ():\n print("Hello, World!")', metadata={}),
 Document( page_content ='# Call the function\ nhello_world ()', metadata={})]
"""
```

**Split by markdownheader**

For example: `md = # Foo\n\n ## Bar\n\ nHi this is Jim \ nHi this is Joe\n\n ## Baz\n\n Hi this is Molly' .`We define the split header : `[("#", "Header 1"),("##", "Header 2")]` The text should be split by the common header, and finally get: \`{'content': 'Hi this is Jim \\nHi this is Joe', 'metadata': {'Header 1': 'Foo', 'Header 2': 'Bar'}} {'content': 'Hi this is Molly', 'metadata': {' Header 1': 'Foo', 'Header 2': 'Baz'}}

```text
from langchain.text_splitter import MarkdownHeaderTextSplitter
markdown_document = "# Foo\n\n ## Bar\n\ nHi this is Jim\n\ nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\ n\n Hi this is Molly"

headers_to_split_on = [
("#", "Header 1"),
("##", "Header 2"),
("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_ splitter. split _text ( markdown_document )
md_header_splits

"""
[ Document( page_content ='Hi this is Jim \ nHi this is Joe', metadata={'Header 1': 'Foo', 'Header 2': 'Bar'}),
     Document( page_content ='Hi this is Lance', metadata={'Header 1': 'Foo', 'Header 2': 'Bar', 'Header 3': 'Boo'}),
     Document( page_content ='Hi this is Molly', metadata={'Header 1': 'Foo', 'Header 2': 'Baz'})]
"""
```

### Model I/O
Language models come in various forms, including Large Language Models (LLMs), chat models, and text embedding models.

- **Large Language Models (LLMs)**: These are the primary models we discuss. They take a text string as input and produce a text string as an output.

- **Chat Models**: These are our second focus. While they are typically powered by a language model, their API is more structured. Specifically, they process a sequence of chat messages as input and generate a corresponding chat message in response.

- **Text Embedding Models**: These models are designed to convert text into a numerical representation. When given text as input, they yield a list of floating-point numbers.

LangChain offers essential tools to seamlessly integrate any language model.

- **TIP: Templating, Dynamic Selection, and Managing Model Input**

A new way to program models is through prompts. A hint refers to the input to the model. This input is usually composed of multiple components. LangChain provides several classes and functions that make it easy to construct and process hint messages. Commonly used methods are: PromptTemplate: where parameterizes model input; and Example Selector: dynamically selects examples to be included in the prompt.

A prompt template can contain: instructions to the language model; a small set of examples to help the language model generate a better response; a question to the language model. For example:

```python3
from langchain import PromptTemplate


template = """/
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate. from_template (template)
prompt. format (product="colorful socks")

"""
You are a naming consultant for new companies.
What is a good name for a company that makes colorful socks?
"""
```

If you need to create a role-related message template, you need to use MessagePromptTemplate .

```python3
template="You are a helpful assistant that translates { input_language } to { output_language }."
system_message_prompt = SystemMessagePromptTemplate.from_template (template)
human_template ="{text}"
human_message_prompt = HumanMessagePromptTemplate. from_template ( human_template )
```

Types of MessagePromptTemplate : LangChain provides different types of MessagePromptTemplate . The most commonly used ones are AIMessagePromptTemplate , SystemMessagePromptTemplate , and HumanMessagePromptTemplate , which create AI messages, system messages, and human messages, respectively.

Generic Prompt Template: Suppose we want LLM to generate an English interpretation of a function name. To accomplish this, we'll create a custom prompt template that takes a function name as input and format the prompt template to provide the source code for that function.

We first create a function that will return the source code of the given function.

```text
import inspect


def get_source_code ( function_name ):
# Get the source code of the function
return inspect. getsource ( function_name )
```

Next, we'll create a custom prompt template that takes the function name as input and formats the prompt template to provide the function's source code.

```python3
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel , validator


class FunctionExplainerPromptTemplate ( StringPromptTemplate , BaseModel ):
"""A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function."""

@validator("input_variables")
def validate_input_variables ( cls , v):
"""Validate that the input variables are correct."""
if len (v ) ! = 1 or " function_name " not in v:
raise ValueError ( " function_name must be the only input_variable .")
return v

def format( self, ** kwargs ) -> str:
# Get the source code of the function
        source_code = get_source_code ( kwargs [" function_name "])

# Generate the prompt to be sent to the language model
prompt = f"""
Given the function name and source code, generate an English language explanation of the function.
Function Name: { kwargs [" function_name " ]._ _name__}
Source Code:
{ source_code }
Explanation:
"""
return prompt

def_prompt_type ( self):
return "function-explainer"
```

- **Language Model: Interface for Language Models**

Language models vary, with notable distinctions between LLMs and chat models. In LangChain, LLM denotes a plain text completion model. These models have APIs that accept a string prompt and return a string completion. For instance, OpenAI's GPT-3 is an LLM. Chat models, while often underpinned by LLMs, are tailored for conversations. Their API differs from the plain text completion model, accepting a list of chat messages, tagged with the speaker (typically "system", "AI", or "human"), and returning an "AI" chat message. Both GPT-4 and Anthropic's Claude are chat models.

To facilitate the interchangeability of LLM and chat models, both adopt the foundational language model interface, revealing common methods like "predict" and "pred messages". While it's advisable to use model-specific methods (e.g., LLM's "predict" and chat model's "predict message"), a shared interface is beneficial for versatile applications.

Using an LLM is straightforward: input a string and receive a string completion.

The generate function allows batch calls and richer outputs. For instance:
```text
llm_result = llm. generate (["Tell me a joke", "Tell me a poem"]*15)

len ( llm_result.generations ) _

'''
30
'''
llm_result . generations [0]

'''
[ Generation( text='\n\ nWhy did the chicken cross the road?\n\ nTo get to the other side!'),
     Generation( text='\n\ nWhy did the chicken cross the road?\n\ nTo get to the other side.')]
'''

llm_result.generations [-1 ]

'''
[Generation(text="\n\ nWhat if love neverspeech \n\ nWhat if love never ended\n\ nWhat if love was only a feeling\n\ nI'll never know this love\n\ nIt's not a feeling\n \ nBut it's what we have for each other\n\ nWe just know that love is something strong\n\ nAnd we can't help but be happy\n\ nWe just feel what love is for us\n\ nAnd we love each other with all our heart\n\ nWe just don't know how\n\ nHow it will go\n\ nBut we know that love is something strong\n\ nAnd we'll always have each other\n\ nIn our lives ."),
     Generation( text='\n\ nOnce upon a time\n\ nThere was a love so pure and true\n\ nIt lasted for centuries\n\ nAnd never became stale or dry\n\ nIt was moving and alive\n\ nAnd the heart of the love-ick\n\ nIs still beating strong and true.')]


You can also access provider specific information that is returned. This information is NOT standardized across providers.
'''

llm_result.llm_output

'''
{' token_usage ': {' completion_tokens ': 3903,
' total_tokens ': 4023 ,
' prompt_tokens ' : 120}}
'''
```

- **Output Analyzer: Deciphering Model Outputs**

Output parsers structure language model responses. They primarily implement two methods:

- "Get format specification": This method returns a string containing an indication of how the output of the language model should be formatted.
- "parse": This method takes a string (assumed to be a response from a language model) and parses it into some structure.
- "parsing with hints": This method is optional and takes a string (supposed to be a response from the language model) and a hint (supposed to be the hint that produced such a response) and parses it into some structure . Hints are primarily provided in cases where the OutputParser wants to retry or fix the output in some way, and needs information from the hint to do so.

```python3
from langchain.prompts import PromptTemplate , ChatPromptTemplate , HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI _

from langchain.output_parsers import PydanticOutputParser _
from pydantic import BaseModel , Field, validator
from typing import list


model_name = 'text-davinci-003'
temperature = 0.0
model = OpenAI( model_name = model_name , temperature=temperature)

# Define your desired data structure.
class Joke( BaseModel ):
setup: str = Field( description="question to set up a joke")
punchline: str = Field( description="answer to resolve the joke")
    
# You can add custom validation logic easily with Pydantic .
@validator('setup')
def question_ends_with_question_mark ( cls , field) :
if field[ -1] != '?':
raise ValueError ( "Badly formed question!")
return field

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser ( pydantic_object =Joke)

prompt = PromptTemplate (
template="Answer the user query.\n{ format_ instructions }\ n{query}\n",
    input_variables = ["query"],
    partial_variables ={ " format_instructions ": parser.get_format_instructions ()}
)

# And a query intended to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."
_input = prompt. format _prompt (query= joke_query )

output = model(_ input. to_ string ( ))

parser. parse (output)

    Joke( setup='Why did the chicken cross the road?', punchline='To get to the other side!')
```

### Chain components

For intricate applications that necessitate linking LLMs in series, the Chain component is essential. LangChain offers a Chain interface for such applications. Its primary interface is:
```python3
class Chain( BaseModel , ABC):
"""Base interface that all chains should implement."""

memory: BaseMemory
callbacks: Callbacks

def __call_ _(
self,
inputs: Any,
        return_only_outputs : bool = False,
callbacks: Callbacks = None,
) -> Dict [ str, Any]:
...
```

Chains allow us to join multiple components together to create a single, coherent application. For example, we can create a chain that takes user input, formats it with a PromptTemplate , and passes the formatted response to LLM. We can build more complex chains by combining multiple chains together, or combining chains with other components. LLMChain is the most basic building block chain. It takes a prompt template, formats it with user input, and returns a response from the LLM.

We start by creating a prompt template:

```text
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate (
    input_variables = ["product"],
template="What is a good name for a company that makes {product}?",
)
```

Then, create a very simple chain that will take input from the user, use it to format the prompt, and send it to LLM.

```python3
from langchain.chains import LLMChain
chain = LLMChain ( llm = llm , prompt=prompt)

# Run the chain only specifying the input variable.
print( chain. run ("colorful socks"))

Colorful Toes Co.

'''
If you have multiple variables, you can use a dictionary to enter them all at once.
'''
prompt = PromptTemplate (
    input_variables = [ "company", "product"],
template="What is a good name for {company} that makes {product}?",
)
chain = LLMChain ( llm = llm , prompt=prompt)
print( chain. run ({
'company': "ABC Startup",
'product': "colorful socks"
}))

    Socktopia Colorful Creations.
```

It is also possible to use a chat model in LLMChain :

```python3
from langchain.chat_models import ChatOpenAI _
from langchain.prompts.chat import (
    ChatPromptTemplate ,
    HumanMessagePromptTemplate ,
)
human_message_prompt = HumanMessagePromptTemplate (
prompt = PromptTemplate (
template="What is a good name for a company that makes {product}?",
            input_variables = ["product"],
)
)
chat_prompt_template = ChatPromptTemplate.from_messages ([ human_message_prompt ])
chat = ChatOpenAI (temperature=0.9)
chain = LLMChain ( llm =chat, prompt= chat_prompt_template )
print( chain. run ("colorful socks"))

Rainbow Socks Co.
```

### Chain Serialization

For saving and loading chains, refer to: [Serialization | ️ Langchain](https://python.langchain.com/docs/modules/chains/how_to/serialization).

### In Summary

LangChain offers a suite of components tailored for diverse applications like personal assistants, chatbots, and more. It streamlines the development of advanced language model applications through its modular design. By grasping core concepts like components, chains, and output parsers, users can craft custom solutions for specific needs. LangChain empowers users to harness the full capabilities of language models, paving the way for intelligent, context-aware applications across various domains.
