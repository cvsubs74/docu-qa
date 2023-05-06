from stats import Stats
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.chains import RetrievalQA

dbname = "docuqa.db"


class DocuQA:
    def __init__(self):
        self.stats = Stats(dbname)

    @staticmethod
    def set_page_config():
        st.set_page_config(page_title="DocuQA", layout="wide")

    @staticmethod
    def display_introduction():
        st.header("DocuQA")
        # Intro
        st.markdown("DocuQA is the ultimate tool for anyone who needs to work with "
                    "long PDF documents. Whether you're a busy professional trying to "
                    "extract insights from a report or a student looking for specific information "
                    "in a textbook, this powerful web app has the tools you need to get the "
                    "job done quickly and efficiently. With its intuitive interface, DocuQA is "
                    "the perfect solution for conducting question-answering tasks on long documents."
                    )

    def display_stats(self):
        # Retrieve the document and query counts
        document_count = self.stats.get_document_count()
        query_count = self.stats.get_query_count()
        total_file_size = self.stats.get_total_file_size()
        fig1 = self.draw_bar_graph_for_document_and_query_count(document_count, query_count)
        fig2 = self.draw_bar_graph_for_total_file_size(total_file_size)
        # Display the chart and heading in the Streamlit app
        st.subheader("App Usage Stats")
        st.plotly_chart(fig1)
        # st.plotly_chart(fig2)
        st.markdown(f"<p style='font-size:20px; color:navy'><b>Total Size Processed: {total_file_size:.2f} MB</b></p>",
                    unsafe_allow_html=True)

    @staticmethod
    def draw_bar_graph_for_document_and_query_count(document_count, query_count):
        # Create a pandas DataFrame with the counter data
        data = {'Counter Type': ['Processed Documents', 'Queries Executed'],
                'Count': [document_count, query_count]}
        df = pd.DataFrame(data)
        # Create a bar chart of the counter data using Plotly Express
        fig = px.bar(df, x='Counter Type', y='Count', color='Counter Type',
                     height=400)
        # Add the count values to the x-axis labels
        fig.update_layout(xaxis_tickangle=-0,
                          xaxis_tickfont_size=12,
                          xaxis=dict(tickmode='array',
                                     tickvals=df['Counter Type'],
                                     ticktext=[f"{c:,d}" for c in df['Count']],
                                     title=''),
                          yaxis=dict(range=[0, max(df['Count'])]))
        # fig.update_traces(marker_color=['blue', 'light blue'])
        # Modify the chart size and margins
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            width=500,
            height=300,
        )
        return fig

    @staticmethod
    def draw_bar_graph_for_total_file_size(size):
        # Create a new DataFrame with the total file size data
        data = {'File size (MB)': [''],
                'Value': [size]}
        df = pd.DataFrame(data)
        # Create a new bar chart for the total file size data
        fig = px.bar(df, x='File size (MB)', y='Value', color='File size (MB)',
                     height=100)
        # Update the layout of the new chart to remove the legend and adjust the margins
        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=0, b=20),
            width=500,
            height=300,
        )
        return fig

    def display_review_form(self):
        rating = self.slider_rating_widget()
        comment = st.text_input("Please leave any comments or suggestions for improvement:")
        if st.button("Submit Review", key="submit"):
            if rating > 0:
                self.stats.add_review(rating, comment)
                st.success("Thank you for your review!")
            else:
                st.warning("Please select a star rating before submitting.")

    # Display the review stats
    def display_review_stats(self):
        num_reviews, avg_rating = self.stats.get_review_stats()
        st.subheader('**Average Rating**')
        st.write(f'Average rating: {avg_rating} ({num_reviews} reviews)')
        st.write('★' * int(round(avg_rating)) + '☆' * int(5 - round(avg_rating)))

    @staticmethod
    def slider_rating_widget():
        st.subheader("Rate App")
        rating = st.slider("", 1, 5, 3, key="slider_rating", format="%d",
                           help="Drag the slider to rate the app")
        return rating

    def display_usage_stats(self):
        # Draw a horizontal line
        st.markdown("<hr>", unsafe_allow_html=True)

        # Define the column layout
        col1, col2, col3, col4, col5 = st.columns((2, 2, 2, 1, 2))

        # Display the usage stats in the columns
        with col1:
            self.display_stats()
        with col3:
            self.display_review_form()
            self.display_review_stats()
        with col5:
            self.display_reviews()

    def display_reviews(self):
        """Display all the reviews in a Streamlit app."""
        reviews = self.stats.get_reviews()
        st.subheader("Reviews")
        if not reviews:
            st.write("No reviews yet. Be the first to leave a review!")
        else:
            with st.container():
                reviews_str = ""
                for rating, comment in reviews:
                    # Create a div element with the star rating
                    st.write(comment + ' ' + '★' * int(round(rating)))

    def add_document_count_and_size(self, size):
        self.stats.add_document_count()
        self.stats.add_file_size(size)

    def user_query(self, vector_store, llm):
        # Retrieve query results
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                         retriever=vector_store.as_retriever())
        query = st.text_area(label="Ask a question", placeholder="Your question..",
                             key="text_input", value="")
        if query:
            # Add the query counter
            self.stats.add_query_count()
            # Run the query
            st.write(qa.run(query))
