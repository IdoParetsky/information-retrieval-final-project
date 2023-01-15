# information-retrieval-final-project
Information Retrieval Course's Final Project

We have created a Search Engine that accepts queries and retrieves relevant Wikipedia Articles (English only).

Our Engine calculates relevancy scores according to Title / Body / Anchor Text weights, whether the query tokens appeared in any group-InvertedIndex's Posting-Lists and each group's Cosine Similarity value with regard to the query.

Our Engine runs on a GCP VM Instance, with the Posting Lists downloaded from our GCP Bucket to the Instance for runtime minimization, while the InvertedIndex and Wiki ID, Title, PageRank and PageViews scores are downloaded from the Bucket in each Server Startup.

Instance IP: 104.197.198.70 (port 8080)
Project ID: crypto-lexicon-370515 (Project Name: Information Retrieval)
Bucket ID: 318419512_318510252
GitHub Repo: https://github.com/IdoParetsky/information-retrieval-final-project/tree/master
Git Directory in Google Drive: https://drive.google.com/drive/u/0/folders/1LUf_YLUbEo4Qj1CTqvKgqYiMgHy9Ejgf

Repository's Contents:<br />
    IR_Project.ipynb - Notebook used for managing the Git directory, Computing the InvertedIndex & PageRank, Downloading Pageviews and merging Wiki ID, Title, PageRank and Pageviews as one dictionary<br />
    run_frontend_in_colab_test_gcp.ipynb - Notebook for running the Debug Engine (1000 articles) with Ngrok & optimizing the group weights on the full Search Engine<br />
    Report.docx - A written summary of our work, in text<br />
    Report.pptx - A written summary of our work, in slides<br />
    README.md - This file. provides project structure.<br />
    queries_train.json - Initial given train set<br />
    new_train.json - Corrected given train set<br />
    graphframes.sh - Cluster initialization file<br />
    wikidumps - A directory containing the first Parquet file, for the creation of a Debug Engine's InvertedIndex<br />
    colab_cache:<br />
        colab_bins - Posting Lists' .bin files<br />
        colab_indexes - groups' InvertedIndex .pkl files<br />
        colab_pagerank.csv.gz - Debug Engine's compressed Wiki ID, Title, PageRank & PageViews table.<br />
        colab_search_frontend.py - Search functions and scoring logic for Debug Engine<br />
        id_title_pr_pv_dict_cast.pkl - A pickle file containing a dictionary mapping Debug Engine's Wiki IDs to (Title, PageRank, PageViews). Wiki IDs are cast as int, PageRank and PageViews are casted as floats<br />
        inverted_index_colab.py - InvertedIndex Class for Debug Engine<br />
        ngrok & ngrok-stable-linux-amd64.zip - Files with which we used Ngrok to run the Debug Engine as a server<br />
        pageviews-202108-user.pkl - August 2021 PageViews<br />
    gcp_cache:<br />
        gcp_pagerank_pageviews.csv.gz - Search Engine's compressed Wiki ID, Title, PageRank & PageViews table<br />
        id_title_pr_pv_dict_cast.pkl - A pickle file containing a dictionary mapping Search Engine's Wiki IDs to (Title, PageRank, PageViews). Wiki IDs are cast as int, PageRank and PageViews are casted as floats<br />
        inverted_index_gcp.py - InvertedIndex Class for Search Engine<br />
        pageviews-202108-user.pkl - August 2021 PageViews<br />
        run_frontend_in_gcp.sh - Given Instance creation commands and explanations<br />
        search_frontend.py - Search functions and scoring logic for Search Engine<br />
        startup_script_gcp.sh - Given python dependencies for Instance<br />
       
