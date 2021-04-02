import numpy as np
import pandas as pd
import pandas_flavor as pf
import matplotlib.pyplot as plt

@pf.register_dataframe_method
def story_telling_plot(df):
    survival_rate = (
        df
        .select_columns(["sex", "survived"])
        .groupby(["sex"])
        .mean()
    )

    male_rate = survival_rate.loc["male"]
    female_rate = survival_rate.loc["female"]

    sex_survived_crosstab = (
        df
        .pivot_wider(
            index="sex",
            names_from="survived",
            aggfunc="size"
        )
        .set_index("sex")
    )

    male_pos = np.random.uniform(0, male_rate, sex_survived_crosstab.loc["male", 1])
    male_neg = np.random.uniform(male_rate, 1, sex_survived_crosstab.loc["male", 0])
    female_pos = np.random.uniform(0, female_rate, sex_survived_crosstab.loc["female", 1])
    female_neg = np.random.uniform(female_rate, 1, sex_survived_crosstab.loc["female", 0])

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    np.random.seed(42)

    # Male Stripplot
    ax.scatter(np.random.uniform(-0.3, 0.3, len(male_pos)), male_pos, color='#004c70', edgecolor='lightgray', label='Male(Survived=1)')
    ax.scatter(np.random.uniform(-0.3, 0.3, len(male_neg)), male_neg, color='#004c70', edgecolor='lightgray', alpha=0.2, label='Male(Survived=0)')

    # Female Stripplot
    ax.scatter(1+np.random.uniform(-0.3, 0.3, len(female_pos)), female_pos, color='#990000', edgecolor='lightgray', label='Female(Survived=1)')
    ax.scatter(1+np.random.uniform(-0.3, 0.3, len(female_neg)), female_neg, color='#990000', edgecolor='lightgray', alpha=0.2, label='Female(Survived=0)')

    # Set Figure & Axes
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-0.03, 1.1)

    # Ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Male', 'Female'], fontweight='bold', fontfamily='serif', fontsize=13)
    ax.set_yticks([], minor=False)
    ax.set_ylabel('')

    # Spines
    for s in ["top","right","left", 'bottom']:
        ax.spines[s].set_visible(False)

    # Title & Explanation
    fig.text(0.1, 1, 'Distribution of Survivors by Gender', fontweight='bold', fontfamily='serif', fontsize=15)    
    fig.text(0.1, 0.96, 'As is known, the survival rate for female is high, with 19% of male and 74% of female.', fontweight='light', fontfamily='serif', fontsize=12)
    fig.text(.80, .0, 'Data visualization by: @subinium', ha='center', fontweight='light', fontfamily='serif', fontsize=12)

    ax.legend(loc=(0.8, 0.5), edgecolor='None')
    plt.tight_layout()
    plt.show()