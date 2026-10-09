# Data

The dataset is not stored in this repository (≈230 MB, and it is redistributed under Kaggle's terms).

1. Download **515K Hotel Reviews Data in Europe** from Kaggle:
   https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
2. Unzip it and place the file here, keeping its original name:

```
data/Hotel_Reviews.csv
```

Or with the Kaggle CLI:

```bash
kaggle datasets download -d jiashenliu/515k-hotel-reviews-data-in-europe -p data --unzip
```

Source: reviews scraped from Booking.com, 1,492 hotels, Aug 2015 - Aug 2017 (as described on the dataset page).

Columns used: `Hotel_Name`, `Review_Date`, `Positive_Review`, `Negative_Review`, `Reviewer_Score`.
Columns deliberately **not** used: `Average_Score` (a hotel-level aggregate that includes the review being predicted, so a leakage risk), reviewer word counts (derived from the text), and `Tags`.
