import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Function to extract text from PDF files
def extract_text_from_pdfs(essay_input):
    pdf_reader = PdfReader(essay_input)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

# Function to process user inputs
# Do I need to add a summarisation chain with text splitter?
def get_text_chunks(extracted_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    text_chunks = text_splitter.split_text(extracted_text)
    return text_chunks

# Function to create vector store 
def get_vectorstore(text_chunks):
    openai_api_key = st.secrets["openai_api_key"]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to use LLM to produce analysis - work on this next
def process_input(vectorstore, grade_input, feedback_input):
    openai_api_key = st.secrets["openai_api_key"]
    llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)
    
    # Potential to improve this with stuctured output instructions
    template = """ 
    Your job is to help a student understand essay feedback they have received from their examiner. Your instructures below are delimited by tripple backticks (```).Please follow these instructions carefully before providing your response.  
    
    ``` USER INPUT:
    You will be given three input variables which will give you the following data: 
    Grade input: {grade_input} provides the grade the student received on the essay
    Feedback input: {feedback_input} provides the written feedback from the examiner
    Student essay: {vectorstore} provides text embeddings of the raw text from the student's essay
    
    ``` TONE OF YOUR RESPONSE:
    In your response, I want you to write as if you're speaking to the student, e.g. "You received a grade of {grade_input} and overall your examiner was {feedback_input}"
    
    ``` IDEAS FOR YOUR RESPONSE:
    I want you to analyse the student's essay yourself using the content in {vectorstore} so that you can provide the student with a depth of feedback that goes above what their examiner has provided. Please analyse the content in {vectorstore} in light of the following essay evaluation considerations: 
    
    Citations
    - Check if the essay includes proper citations for all sources used, both in-text and in the bibliography or reference list.
    - Ensure the citations follow a consistent citation style (such as APA, MLA, Chicago, etc.).
    - Verify that direct quotes, paraphrased content, and ideas are appropriately attributed to their sources.
    - Look for a variety of reputable sources that support the arguments and ideas presented in the essay.
    
    Spelling and grammar
    - Assess spelling, grammar, and punctuation errors throughout the essay.
    - Check for subject-verb agreement, tense consistency, and sentence structure.
    - Ensure that the essay is free of typographical mistakes that might affect its readability.
    
    Writing style
    - Evaluate the overall tone and voice of the essay: is it formal, objective, persuasive, or descriptive, as required by the assignment?
    - Consider the clarity and coherence of the writing. Are ideas presented logically and in a well-organized manner?
    - Look for appropriate use of transitions between paragraphs and ideas.
    - Assess the use of varied sentence structures and vocabulary to maintain reader engagement.
    
    Argument strength
    - Examine the clarity and coherence of the main thesis statement or argument.
    - Evaluate the use of evidence to support the argument. Is the evidence relevant, credible, and properly integrated into the essay?
    - Consider counterarguments and whether the essay addresses them effectively.
    - Assess whether the essay provides strong reasoning and logical connections between ideas.
    
    Originality
    - Look for evidence of original thought and critical thinking in the essay.
    - Consider whether the essay brings new perspectives or insights to the topic.
    
    Research quality
    - Evaluate the depth and breadth of research conducted for the essay.
    - Assess whether the essay engages with a variety of viewpoints and sources, showcasing a comprehensive understanding of the subject.
    - Consider whether the research is integrated smoothly into the essay, supporting the argument without overwhelming the text.
    
    ``` STRUCTURE OF YOUR RESPONSE:
    Please follow this structured format when giving your response:

    1. **Grade Summary**: Provide a concise summary of the reasons why the student received the given grade, e.g. "You received a grade of {grade_input} because..." 

    2. **Strengths**: List the strengths identified in the student's essay based on the feedback, e.g. "You did well in..."

    3. **Areas for Improvement**: Enumerate the specific areas where the student can improve, addressing the feedback points, e.g. "Your examiner believes that an area for improvement is..."

    4. **Recommendations**: Offer constructive recommendations and guidance on how the student can enhance their performance in future essays, e.g. "In the future, consider..."
    
    ``` YOUR RESPONSE:
    
    """
    
    prompt = PromptTemplate(
        input_variables=["grade_input", "feedback_input", "vectorstore"],
        template=template,
    )
    
    final_prompt = prompt.format(grade_input=grade_input, feedback_input=feedback_input, vectorstore=vectorstore)
    analysis_results = llm(final_prompt)
    
    return analysis_results


# Function to run Streamlit frontend
def main():
    st.set_page_config(page_title="Lammy",
                       page_icon=":lower_left_ballpoint_pen:")
    
    # Setting up containers
    header = st.container()
    essay_upload = st.container()
    feedback_upload = st.container()
    #output = st.container() # Need to divide up output containers at later stage to align with wireframes
    
    with header:
        st.header("Lammy")
        st.write("Feedback, Upgraded: Get Actionable Insights with Lammy :lower_left_ballpoint_pen:")
        
    with essay_upload:
        st.markdown("#### Essay")
        st.file_uploader(label="Upload your essay (PDF)", key="essay_input")
    
    with feedback_upload:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Grade")
            feedback_upload = st.text_area(label="Enter your grade out of 100, e.g. 68/100", key="grade_input")
            
        with col2:
            st.markdown("#### Feedback")
            feedback_upload = st.text_area(label="Copy and paste your feedback", key="feedback_input")
    
    # Function to retrieve user input on click of submit button
    def on_submit():
        essay_input = st.session_state.essay_input  # Get essay input
        grade_input = st.session_state.grade_input.strip()     # Get grade input
        feedback_input = st.session_state.feedback_input.strip()  # Get feedback input
        
        # Check if essay was uploaded
        if not essay_input:
            st.warning("Please try uploading again essay.")
            return
        
        # Extract text from essay
        extracted_text = extract_text_from_pdfs(essay_input)
        
        # Get text chunks from essay
        text_chunks = get_text_chunks(extracted_text)
        
        # Create vector store
        vectorstore = get_vectorstore(text_chunks)
        
        # Process results
        analysis_result = process_input(vectorstore, grade_input, feedback_input)
        
        # Display the analysis result
        st.markdown("#### Analysis:")
        st.write(analysis_result)
        
    if st.button("Submit", key="submit_button"):
        with st.spinner("Processing"):
            on_submit()  # Call the on_submit function

if __name__ == '__main__':
     main()
