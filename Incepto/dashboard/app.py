import shutil
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit import caching
import torchvision
from torchray.attribution.deconvnet import deconvnet
import torchray
from torchray.benchmark import get_example_data, plot_example
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.guided_backprop import guided_backprop
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.gradient import gradient
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.attribution.linear_approx import linear_approx
from torchray.attribution.rise import rise
from flashtorch.activmax import GradientAscent

import torch
from matplotlib import pyplot as plt
import cv2
import matplotlib as mpl

warnings.filterwarnings("ignore")
st.set_option('deprecation.showfileUploaderEncoding', False)

def get_layers(model):
    layers=[]
    for name, param in model.named_parameters():
        name = name.replace("weight","bias")
        name = name.split(".bias")[0]
        if(name not in layers):
            layers.append(name)
    return layers

def app(model = torchvision.models.resnet18().eval(), in_dist_name="in", ood_data_names=["out","out2"], image=True):
    # Render the readme as markdown using st.markdown.
    st.markdown(get_file_content_as_string("Incepto/dashboard/intro.md"))
    layers = get_layers(model)
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    if st.sidebar.button("Go to Guide"):
        caching.clear_cache()
        st.markdown(get_file_content_as_string("Incepto/dashboard/details.md"))

    st.sidebar.title("Data Settings")
    # select which set of SNPs to explore

    dataset = st.sidebar.selectbox(
        "Set Dataset:",
        (in_dist_name,*ood_data_names),
    )
    
    if image:
        visualization = st.sidebar.selectbox(
            "Set Visualization Type:",
            ("-", "Color Distribution for Entire Dataset", "Pixel Distribution for Entire Dataset", "Deconvolution", "Excitation Backpropgation","Gradient","Grad-CAM","Guided Backpropagation","Linear Approximation", "Extremal Perturbation", "RISE"),
        )
    else:
        visualization = st.sidebar.selectbox(
            "Set Visualization Type:",
            ("-", "Average Signal for Entire Dataset", "Deconvolution", "Excitation Backpropgation","Gradient","Grad-CAM","Guided Backpropagation","Linear Approximation", "Extremal Perturbation", "RISE"),
        )

    if image:
        if visualization == "Deconvolution":
                caching.clear_cache()
                saliency_layer=st.selectbox("Select Layer:",tuple(layers))
                # st.number_input(label="Enter a channel number:", step=1, min_value=0, value=0)
                _, x, category_id, _ = get_example_data()
                saliency = deconvnet(model, x.cpu(), category_id, saliency_layer=saliency_layer)
                fig = plt.figure(figsize=(40,40))
                ax = fig.add_subplot(131)
                ax.imshow(np.asarray(saliency.squeeze()))
                ax = fig.add_subplot(132)
                ax.imshow(np.asarray(x.cpu().squeeze().permute(1,2,0).detach().numpy() ))
                st.pyplot(fig)
        elif visualization == "Grad-CAM":
            with st.spinner("Generating Plot"):
                caching.clear_cache()
                saliency_layer=st.selectbox("Select Layer:",tuple(layers))
                # st.number_input(label="Enter a channel number:", step=1, min_value=0, value=0)
                _, x, category_id, _ = get_example_data()
                saliency = linear_approx(model, x.cpu(), category_id, saliency_layer=saliency_layer)
                fig = plt.figure(figsize=(40,40))
                ax = fig.add_subplot(131)
                ax.imshow(np.asarray(saliency.squeeze().detach().numpy() ))
                ax = fig.add_subplot(132)
                ax.imshow(np.asarray(x.cpu().squeeze().permute(1,2,0).detach().numpy() ))
                st.pyplot(fig)
        elif visualization == "Guided Backpropagation":
            with st.spinner("Generating Plot"):
                caching.clear_cache()
                saliency_layer=st.selectbox("Select Layer:",tuple(layers))
                # st.number_input(label="Enter a channel number:", step=1, min_value=0, value=0)
                _, x, category_id, _ = get_example_data()
                saliency = guided_backprop(model, x.cpu(), category_id, saliency_layer=saliency_layer)
                fig = plt.figure(figsize=(40,40))
                ax = fig.add_subplot(131)
                ax.imshow(np.asarray(saliency.squeeze().detach().numpy() ))
                ax = fig.add_subplot(132)
                ax.imshow(np.asarray(x.cpu().squeeze().permute(1,2,0).detach().numpy() ))
            st.pyplot(fig)
        elif visualization == "Gradient":
            with st.spinner("Generating Plot"):
                caching.clear_cache()
                saliency_layer=st.selectbox("Select Layer:",tuple(layers))
                # st.number_input(label="Enter a channel number:", step=1, min_value=0, value=0)
                _, x, category_id, _ = get_example_data()
                saliency = gradient(model, x.cpu(), category_id, saliency_layer=saliency_layer)
                fig = plt.figure(figsize=(40,40))
                ax = fig.add_subplot(131)
                ax.imshow(np.asarray(saliency.squeeze().detach().numpy() ))
                ax = fig.add_subplot(132)
                ax.imshow(np.asarray(x.cpu().squeeze().permute(1,2,0).detach().numpy() ))
            st.pyplot(fig)
        elif visualization == "Linear Approximation":
            with st.spinner("Generating Plot"):
                caching.clear_cache()
                saliency_layer=st.selectbox("Select Layer:",tuple(layers))
                # st.number_input(label="Enter a channel number:", step=1, min_value=0, value=0)
                _, x, category_id, _ = get_example_data()
                saliency = gradient(model, x.cpu(), category_id, saliency_layer=saliency_layer)
                fig = plt.figure(figsize=(40,40))
                ax = fig.add_subplot(131)
                ax.imshow(np.asarray(saliency.squeeze().detach().numpy() ))
                ax = fig.add_subplot(132)
                ax.imshow(np.asarray(x.cpu().squeeze().permute(1,2,0).detach().numpy() ))
            st.pyplot(fig)
        elif visualization == "Extremal Perturbation":
            with st.spinner("Generating Plot"):
                caching.clear_cache()
                # saliency_layer=st.selectbox("Select Layer:",tuple(layers))
                # st.number_input(label="Enter a channel number:", step=1, min_value=0, value=0)
                _, x, category_id, _ = get_example_data()
                masks_1, _ = extremal_perturbation(
                    model, x.cpu(), category_id,
                    reward_func=contrastive_reward,
                    debug=False,
                    areas=[0.12],)
                fig = plt.figure(figsize=(40,40))
                ax = fig.add_subplot(131)
                ax.imshow(np.asarray(masks_1.squeeze().detach().numpy() ))
                ax = fig.add_subplot(132)
                ax.imshow(np.asarray(x.cpu().squeeze().permute(1,2,0).detach().numpy() ))
                st.pyplot(fig)
        elif visualization == "RISE":
            with st.spinner("Generating Plot"):
                caching.clear_cache()
                # saliency_layer=st.selectbox("Select Layer:",tuple(layers))
                # st.number_input(label="Enter a channel number:", step=1, min_value=0, value=0)
                _, x, category_id, _ = get_example_data()
                saliency = rise(model, x.cpu())
                saliency = saliency[:, category_id].unsqueeze(0)
                fig = plt.figure(figsize=(40,40))
                ax = fig.add_subplot(131)
                ax.imshow(np.asarray(saliency.squeeze().detach().numpy() ))
                ax = fig.add_subplot(132)
                ax.imshow(np.asarray(x.cpu().squeeze().permute(1,2,0).detach().numpy() ))
                st.pyplot(fig)
        elif visualization == "Color Distribution for Entire Dataset":
            with st.spinner("Generating Plot"):
                caching.clear_cache()
                # saliency_layer=st.selectbox("Select Layer:",tuple(layers))
                # st.number_input(label="Enter a channel number:", step=1, min_value=0, value=0)
                _, x, category_id, _ = get_example_data()
                x = sum(x)/len(x)
                image = x.cpu().detach().numpy()
                fig = plt.figure()
                mpl.rcParams.update({'font.size': 15})
                _ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
                _ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
                _ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
                _ = plt.xlabel('Intensity Value')
                _ = plt.ylabel('Count')
                _ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
                
                st.pyplot(fig)
        elif visualization == "Pixel Distribution for Entire Dataset":
            with st.spinner("Generating Plot"):
                caching.clear_cache()
                
                _, x, category_id, _ = get_example_data()
                x = sum(x)/len(x)
                image = x.cpu().detach().numpy()
                fig = plt.figure(figsize=(40,40))
                plt.ylabel("Count")
                plt.xlabel("Intensity Value")
                mpl.rcParams.update({'font.size': 55})

                ax = plt.hist(x.cpu().detach().numpy().ravel(), bins = 256)
                vlo = cv2.Laplacian(x.cpu().detach().numpy().ravel(), cv2.CV_32F).var()
                plt.text(1, 1, ('Variance of Laplacian: '+str(vlo)), style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
                st.pyplot(fig)
            mpl.rcParams.update({'font.size': 15})
            
    if st.sidebar.button("Visualize Model"):
        saliency_layer=st.selectbox("Select Layer:",tuple(layers))
        filter = st.number_input(label="Enter a filter number:", step=1, min_value=1, value=1)
        g_ascent = GradientAscent(model)
        g_ascent.use_gpu = False
        layer = model.conv1
        exec("layer = model.conv1")
        print(layer)
        img = g_ascent.visualize(layer, filter, title=saliency_layer,return_output=True)[0][0][0]
        fig = plt.figure(figsize=(40,40))
        ax = fig.add_subplot(131)
        ax.imshow(np.asarray(img.cpu().detach().numpy() ))
        st.pyplot(fig)
        
        
        

    # upload the file
    # user_file = st.sidebar.file_uploader("Upload your model in .pth format:")

    # if user_file is not None:
    #     try:
    #         with st.spinner("Uploading your model..."):
    #             with open("user_snps_file.txt", "w") as file:
    #                 user_file.seek(0)
    #                 shutil.copyfileobj(user_file, file)
    #     except Exception as e:
    #         st.error(
    #             f"Sorry, there was a problem processing your model file.\n {e}"
    #         )


    # filter and encode the user record
    #     user_record, aisnps_1kg = filter_user_genotypes(userdf, aisnps_1kg)
    #     user_encoded = encoder.transform(user_record)
    #     X_encoded = np.concatenate((X_encoded, user_encoded))
    #     del userdf
    #
    #     # impute the user record and reduce the dimensions
    #     user_imputed = impute_missing(X_encoded)
    #     user_reduced = reducer.transform([user_imputed])
    #     # fit the knn before adding the user sample
    #     knn.fit(X_reduced, dfsamples[population_level])
    #
    #     # concat the 1kg and user reduced arrays
    #     X_reduced = np.concatenate((X_reduced, user_reduced))
    #     dfsamples.loc["me"] = ["me"] * 3
    #
    #     # plot
    #     plotly_3d = plot_3d(X_reduced, dfsamples, population_level)
    #     st.plotly_chart(plotly_3d, user_container_width=True)
    #
    #     # predict the population for the user sample
    #     user_pop = knn.predict(user_reduced)[0]
    #     st.subheader(f"Your predicted {population_level}")
    #     st.text(f"Your predicted population using KNN classifier is {user_pop}")
    #     # show the predicted probabilities for each population
    #     st.subheader(f"Your predicted {population_level} probabilities")
    #     user_pop_probs = knn.predict_proba(user_reduced)
    #     user_probs_df = pd.DataFrame(
    #         [user_pop_probs[0]], columns=knn.classes_, index=["me"]
    #     )
    #     st.dataframe(user_probs_df)
    #
    #     show_user_gts = st.sidebar.checkbox("Show Your Genotypes")
    #     if show_user_gts:
    #         user_table_title = "Genotypes of Ancestry-Informative SNPs in Your Sample"
    #         st.subheader(user_table_title)
    #         st.dataframe(user_record)
    #
    # else:
    #     # plot
    #     plotly_3d = plot_3d(X_reduced, dfsamples, population_level)
    #     st.plotly_chart(plotly_3d, user_container_width=True)

    # Collapsable 1000 Genomes sample table
    # show_1kg = st.sidebar.checkbox("Show 1k Genomes Genotypes")
    # if show_1kg is True:
    #     table_title = (
    #         "Genotypes of Ancestry-Informative SNPs in 1000 Genomes Project Samples"
    #     )
    #     with st.spinner("Loading 1k Genomes DataFrame"):
    #         st.subheader(table_title)
    #         st.dataframe(aisnps_1kg)


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
