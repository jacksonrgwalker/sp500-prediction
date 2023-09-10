from glob import glob
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pylcs
from dask.distributed import Client
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# mark as matched if it's the best match for the keyword
# and the similarity is better or equal to this threshold
DEFAULT_MATCH_THRESHOLD = 0.9

# best matches that are better than this threshold
# but worse than DEFAULT_MATCH_THRESHOLD will be
# marked as recommended for manual review
CHECK_MATCH_THRESHOLD = 0.8

# if the similarity is below this threshold, it's not a match
NON_MATCH_THRESHOLD = 0.5

# The top number of matches to keep for each keyword
KEEP_BEST_N = 5

###########################################################################
###########################################################################
###########################################################################

if __name__ == "__main__":
    print(f"Processing NYT keywords...")

    # read in all the NYT article parquet files
    nyt_files = glob("data/NYT/*/*.parquet")
    nyt_articles = pd.concat([pd.read_parquet(f) for f in nyt_files])
    nyt_articles.reset_index(drop=True, inplace=True)

    # extract the keywords from the nested NYT API response
    keyword_nested = nyt_articles.set_index("_id")["keywords"].explode().dropna()
    keywords = pd.json_normalize(keyword_nested)
    # json_normalize drops the index, so we need to add it back
    keywords = keywords.set_index(keyword_nested.index)

    keywords.drop(columns=["rank", "major"], inplace=True)
    keywords.rename(columns={"name": "type", "value": "keyword"}, inplace=True)

    keywords["type"] = keywords["type"].astype("category")
    keep_keyword_types = ["organizations", "subject"]
    keywords = keywords[keywords["type"].isin(keep_keyword_types)].copy()

    # create an initial unique id for each keyword
    keywords["keyword_id"] = pd.factorize(keywords["keyword"])[0]

    # keep track of which article each keyword came from
    # multiple articles can have the same keyword
    article_to_keyword_map = keywords["keyword_id"].copy()

    # now we can drop duplicate keywords
    keywords.drop_duplicates(subset=["keyword_id"], inplace=True)
    keywords.set_index("keyword_id", inplace=True)

    def preprocess_keywords(string_series):
        """
        Function to preprocess the keywords and company names
        before calculating the similarity.

        The preprocessing steps are:
            1. Convert all strings to lowercase.
            2. Remove text that is in parentheses.
            3. Replace "&" with "and".
            4. Remove all punctuation, except for "-".
            5. Lemmatize the text using the WordNetLemmatizer from the Natural Language Toolkit (NLTK).
            6. Replace common synonyms for words such as "corporation", "world", "association", "school", and "technology".
            7. Remove leading and trailing whitespace.
            8. Replace all whitespace with a single space.
        """
        # lowercase
        string_series = string_series.str.lower()
        # remove text that is in parantheses
        string_series = string_series.str.replace(r"\(.*\)", "", regex=True)
        # replace "&" with "and"
        string_series = string_series.str.replace("&", " and ")
        # remove punctuation, except for "-"
        string_series = string_series.str.replace(r"[^\w\s-]", "", regex=True)

        wnl = WordNetLemmatizer()

        def lemmatize_text(text):
            return " ".join([wnl.lemmatize(word) for word in word_tokenize(text)])

        string_series = string_series.apply(lemmatize_text)

        corp_synonyms = [
            "corporation",
            "incorporated",
            "bancorporation",
            "company",
            "companies",
            "corp",
            "inc",
            "plc",
            "co",
            "limited",
            "ltd",
            "llc",
            "lp",
            "holding",
            "holdings",
            "group",
            "groups",
            "partnership",
            "trust",
        ]

        world_synonyms = [
            "global",
            "international",
            "national",
            "worldwide",
            "world",
        ]

        association_synonyms = [
            "association",
            "associate",
            "union",
            "center",
            "institute",
        ]

        tech_synonyms = [
            "technology",
            "tech",
        ]

        school_synonyms = [
            "university",
            "college",
            "school",
            "academy",
        ]

        synonyms = [
            ("inc", corp_synonyms),
            ("world", world_synonyms),
            ("assn", association_synonyms),
            ("school", school_synonyms),
            ("tech", tech_synonyms),
        ]

        for word, syns in synonyms:
            for syn in syns:
                string_series = string_series.str.replace(
                    rf"\b{syn}\b", word, regex=True
                )
                # also replace the rough plural
                string_series = string_series.str.replace(
                    rf"\b{syn}s\b", f"{word}", regex=True
                )

        # remove leading and trailing whitespace
        string_series = string_series.str.strip()
        # make all whitespace a single space
        string_series = string_series.str.replace(r"\s+", " ", regex=True)

        return string_series

    keywords["keyword_norm"] = preprocess_keywords(keywords["keyword"])

    # filter out keywords types that are not related to the S&P 500 or constituents
    is_sp500 = keywords["keyword_norm"].str.contains("standard and poor")
    unrelated = (keywords["type"] == "subject") & ~(is_sp500)
    keywords = keywords[~unrelated].copy()

    keywords["keyword_norm_id"] = pd.factorize(keywords["keyword_norm"])[0]

    # keep track of which keyword each keyword_norm came from
    # multiple keywords can have the same keyword_norm
    keyword_to_keyword_norm_map = keywords["keyword_norm_id"].copy()

    # now we can drop duplicate keywords again after normalization
    keywords.drop_duplicates(subset=["keyword_norm_id"], inplace=True)
    keywords.set_index("keyword_norm_id", inplace=True)
    keyword_article_mapper = article_to_keyword_map.map(keyword_to_keyword_norm_map)
    keyword_article_mapper = (
        keyword_article_mapper.dropna().astype(int).rename("keyword_norm_id")
    )

    print(f"Processing S&P 500 symbols...")

    # the symbols table contains the company names and ticker symbols
    symbols = pd.read_parquet(Path("data/symbols.parquet"))
    symbols["company_name_norm"] = preprocess_keywords(symbols["security"])

    # candidate matches are all combinations of keywords and company names
    left = keywords["keyword_norm"].reset_index()
    right = symbols[["symbol", "company_name_norm"]]
    candidates = left.merge(right, how="cross")

    ##### START DASK SECTION ###################################
    print(f"Using dask to compare {len(candidates):,} candidates...")

    # using dask to parallelize the calculation
    # start distributed scheduler locally.
    with Client() as client:
        # convert to dask dataframe for parallel processing
        candidates_dd = dd.from_pandas(candidates, npartitions=1000).reset_index(
            drop=True
        )

        # calculate the longest common substring between the normalized keyword and company name
        # using the pylcs package, which has a fast C implementation
        candidates_dd["lcs_len"] = candidates_dd.apply(
            lambda x: pylcs.lcs_string_length(
                x["keyword_norm"], x["company_name_norm"]
            ),
            axis=1,
            meta=("lcs_len", "int64"),
        )

        # divide by the length of the original strings to get an unbiased measure of similarity
        # this is how the similarity is calculated in the original paper
        candidates_dd["lcs_kwd_pct"] = (
            candidates_dd["lcs_len"] / candidates_dd["keyword_norm"].str.len()
        )
        candidates_dd["lcs_cmp_pct"] = (
            candidates_dd["lcs_len"] / candidates_dd["company_name_norm"].str.len()
        )
        candidates_dd["similarity"] = (
            candidates_dd["lcs_kwd_pct"] + candidates_dd["lcs_cmp_pct"]
        ) / 2

        # filter out obvious non-matches
        candidates_dd = candidates_dd[candidates_dd["similarity"] > NON_MATCH_THRESHOLD]

        matches_dd = candidates_dd.groupby("keyword_norm_id").apply(
            lambda x: x.nlargest(KEEP_BEST_N, "similarity")[
                [
                    "keyword_norm",
                    "similarity",
                    "company_name_norm",
                    "symbol",
                ]
            ],
            meta={
                "keyword_norm": "string",
                "similarity": "float64",
                "company_name_norm": "string",
                "symbol": "string",
            },
        )

        # trigger computation to get pandas dataframe back
        matches = matches_dd.compute()

    print(f"Done calculating matches with dask")
    ##### END DASK SECTION ###################################

    print(f"Processing output...")

    matches = matches.reset_index(level=0).reset_index(drop=True)

    # join back in original keyword and company name
    matches = matches.merge(
        keywords["keyword"], right_index=True, left_on="keyword_norm_id"
    )
    matches = matches.merge(
        symbols[["security", "symbol"]], left_on="symbol", right_on="symbol", how="left"
    )
    matches.rename(
        columns={
            "keyword": "keyword_original",
            "security": "company_name_original",
        },
        inplace=True,
    )

    # get rank of each match for each keyword
    matches["similarity_rank"] = matches.groupby("keyword_norm_id")[
        "similarity"
    ].transform(lambda x: x.rank(ascending=False))
    matches["similarity_rank"] = (
        matches["similarity_rank"].astype(int).astype(pd.Int8Dtype())
    )

    # mark as matched if it's the best match for the keyword
    # and the similarity is above a threshold
    is_best_match = matches["similarity_rank"] == 1
    match_threshold = matches["similarity"] >= DEFAULT_MATCH_THRESHOLD
    matches["default_match"] = is_best_match & match_threshold

    # Mark which records we recommend checking manually
    # and possibly overriding the default match
    check_threshold = matches["similarity"] >= CHECK_MATCH_THRESHOLD
    matches["review_recomended"] = is_best_match & check_threshold & ~match_threshold

    matches["default_match_override"] = ""

    print(f"Marked {matches['default_match'].sum()} matches as default matches")
    print(
        f"Marked {matches['review_recomended'].sum()} matches as recommended for manual review"
    )

    keyword_article_mapper.to_frame().to_parquet("data/keyword_article_mapper.parquet")
    matches.to_csv("data/match_review.csv", index=False)
