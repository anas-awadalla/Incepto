import warnings
import csv
import numpy as np
from matplotlib import pyplot as plt
from streamlit import caching
import streamlit as st
import collections
import functools
import inspect
import textwrap

warnings.filterwarnings("ignore")
result = []
global index


def get_index():
    index += 1
    return index


def cache_on_button_press(label, **cache_kwargs):
    internal_cache_kwargs = dict(cache_kwargs)
    internal_cache_kwargs['allow_output_mutation'] = True
    internal_cache_kwargs['show_spinner'] = False

    def function_decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            @st.cache(**internal_cache_kwargs)
            def get_cache_entry(func, args, kwargs):
                class ButtonCacheEntry:
                    def __init__(self):
                        self.evaluated = False
                        self.return_value = None

                    def evaluate(self):
                        self.evaluated = True
                        self.return_value = func(*args, **kwargs)

                return ButtonCacheEntry()

            cache_entry = get_cache_entry(func, args, kwargs)
            if not cache_entry.evaluated:
                if st.button(label):
                    cache_entry.evaluate()
                else:
                    raise st.StopException
            return cache_entry.return_value

        return wrapped_func

    return function_decorator


def app():
    index = 0

    # hide_streamlit_style = """
    #             <style>
    #             #MainMenu {visibility: hidden;}
    #             footer {visibility: hidden;}
    #             </style>
    #             """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # Render the readme as markdown using st.markdown.

    def next_images(index=0):
        if index == 0:
            st.text("Confidence: 25%")
            st.text("Result: Cancerous Tumor")
            st.text("")
            st.image("Incepto/user-testing-app/images/1-original.jpeg")
            st.image("Incepto/user-testing-app/images/1-mask.png")
        elif index == 1:
            st.text("Confidence: 5%")
            st.text("Result: Non-cancerous Tumor")
            st.text("")
            st.image("Incepto/user-testing-app/images/2-original.jpg")
            st.image("Incepto/user-testing-app/images/2-mask.png")
        else:
            return True

    st.markdown(get_file_content_as_string("Incepto/user-testing-app/intro.md"))
    st.video(None)
    started = False
    start = st.button("Start Testing")
    finished = True

    st.sidebar.title("Testing Menu")
    # response = st.sidebar.radio("Choose Your Response", ["Trust Model", "Model Not Trusted"])
    # next_case = st.sidebar.button("Submit Response For this Case")
    # submit = st.sidebar.button("Finish Testing")

    if start:
        started = True
        next_images(index)

    if st.sidebar.button("Trust Model"):
        index += 1
        result.append(1)
        end = next_images(index)
        if end:
            st.success("Thank you for completing the survey! Please click submit below to save responses")

    if st.sidebar.button("Do Not Trust Model"):
        index += 2
        result.append(0)
        end = next_images(index)
        if end:
            st.sidebar.success("Thank you for completing the survey! Please click Finish Testing to save responses")

    # if st.sidebar.button("Submit Response For this Case"):
    #     index += 1
    #     end = next_images(index)
    #     if end:
    #         st.success("Thank you for completing the survey! Please click submit below to save responses")
    #
    # st.sidebar.empty()
    if st.sidebar.button("Finish Testing"):
        # if finished and started:
        with open('testing.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(result)

        st.success("Your results have been stored.")
        # else:
        #     st.warning("You have not finished the testing yet!")


@st.cache
def get_file_content_as_string(mdfile):
    """Convenience function to convert file to string

    :param mdfile: path to markdown
    :type mdfile: str
    :return: file contents
    :rtype: str
    """
    mdstring = ""
    with open(mdfile, "r") as f:
        for line in f:
            mdstring += line
    return mdstring


if __name__ == "__main__":
    app()
