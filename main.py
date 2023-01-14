import cohere
import streamlit as st

st.write("HELLO WORLD")

co = cohere.Client('knET4NqdcT9EKSuN0tkxYpdrxHOu2MutKZ7n7IRk')
response = co.generate(
    model='xlarge',
    prompt='This program generates mental health coping mechanisms from simple commands given an problem. Here are some examples:\nCommand: I am depressed\nEmotion: depression\nResponse: When feeling depressed, go eat something sweet\n--\nCommand: I am suicidal\nEmotion: suicide\nResponse: Let\'s start off by listing the things you are greatful for\n--\nCommand: I am having an anxiety attack\nEmotion: anxiety\nResponse: Start by taking some deep breaths. \n--\nCommand: I am panicking\nEmotion: panic\nResponse: Imagine you are somewhere calm, like the beach.\n--\nCommand: I am nervous\nEmotion: nervous\nResponse: Hold your hands under cold water\n--\nCommand: I need strategies to help with my depression',
    max_tokens=50,
    temperature=0.9,
    k=0,
    p=0.75,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=[],
    return_likelihoods='NONE')
print('Prediction: {}'.format(response.generations[0].text))
st.write('Prediction: {}'.format(response.generations[0].text))
