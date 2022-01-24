## Week 1 Homework

In this homework we'll prepare the environment 
and practice with terraform and SQL

## Question 1. Google Cloud SDK

Install Google Cloud SDK. What's the version you have?

To get the version, run `gcloud --version`

**Answer:** Google Cloud SDK 369.0.0

## Google Cloud account

Create an account in Google Cloud and create a project.

## Question 2. Terraform

Now install terraform and go to the terraform directory (`week_1_basics_n_setup/1_terraform_gcp/terraform`)

After that, run

* `terraform init`
* `terraform plan`
* `terraform apply` 

Apply the plan and copy the output (after running `apply`) to the form

**Answer:**

```bash
$ terraform init

Initializing the backend...

Initializing provider plugins...
- Reusing previous version of hashicorp/google from the dependency lock file
- Using previously-installed hashicorp/google v4.7.0

Terraform has been successfully initialized!

You may now begin working with Terraform. Try running "terraform plan" to see
any changes that are required for your infrastructure. All Terraform commands
should now work.

If you ever set or change modules or backend configuration for Terraform,
rerun this command to reinitialize your working directory. If you forget, other
commands will detect it and remind you to do so if necessary.

$ terraform plan
var.project
  Your GCP Project ID

  Enter a value: keen-philosophy-339115


Terraform used the selected providers to generate the following execution plan. Resource actions are indicated with the following
symbols:
  + create

Terraform will perform the following actions:

  # google_bigquery_dataset.dataset will be created
  + resource "google_bigquery_dataset" "dataset" {
      + creation_time              = (known after apply)
      + dataset_id                 = "trips_data_all"
      + delete_contents_on_destroy = false
      + etag                       = (known after apply)
      + id                         = (known after apply)
      + last_modified_time         = (known after apply)
      + location                   = "europe-west6"
      + project                    = "keen-philosophy-339115"
      + self_link                  = (known after apply)

      + access {
          + domain         = (known after apply)
          + group_by_email = (known after apply)
          + role           = (known after apply)
          + special_group  = (known after apply)
          + user_by_email  = (known after apply)

          + view {
              + dataset_id = (known after apply)
              + project_id = (known after apply)
              + table_id   = (known after apply)
            }
        }
    }

  # google_storage_bucket.data-lake-bucket will be created
  + resource "google_storage_bucket" "data-lake-bucket" {
      + force_destroy               = true
      + id                          = (known after apply)
      + location                    = "EUROPE-WEST6"
      + name                        = "dtc_data_lake_keen-philosophy-339115"
      + project                     = (known after apply)
      + self_link                   = (known after apply)
      + storage_class               = "STANDARD"
      + uniform_bucket_level_access = true
      + url                         = (known after apply)

      + lifecycle_rule {
          + action {
              + type = "Delete"
            }

          + condition {
              + age                   = 30
              + matches_storage_class = []
              + with_state            = (known after apply)
            }
        }

      + versioning {
          + enabled = true
        }
    }

Plan: 2 to add, 0 to change, 0 to destroy.

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Note: You didn't use the -out option to save this plan, so Terraform can't guarantee to take exactly these actions if you run
"terraform apply" now.

$ terraform apply
var.project
  Your GCP Project ID

  Enter a value: keen-philosophy-339115


Terraform used the selected providers to generate the following execution plan. Resource actions are indicated with the following
symbols:
  + create

Terraform will perform the following actions:

  # google_bigquery_dataset.dataset will be created
  + resource "google_bigquery_dataset" "dataset" {
      + creation_time              = (known after apply)
      + dataset_id                 = "trips_data_all"
      + delete_contents_on_destroy = false
      + etag                       = (known after apply)
      + id                         = (known after apply)
      + last_modified_time         = (known after apply)
      + location                   = "europe-west6"
      + project                    = "keen-philosophy-339115"
      + self_link                  = (known after apply)

      + access {
          + domain         = (known after apply)
          + group_by_email = (known after apply)
          + role           = (known after apply)
          + special_group  = (known after apply)
          + user_by_email  = (known after apply)

          + view {
              + dataset_id = (known after apply)
              + project_id = (known after apply)
              + table_id   = (known after apply)
            }
        }
    }

  # google_storage_bucket.data-lake-bucket will be created
  + resource "google_storage_bucket" "data-lake-bucket" {
      + force_destroy               = true
      + id                          = (known after apply)
      + location                    = "EUROPE-WEST6"
      + name                        = "dtc_data_lake_keen-philosophy-339115"
      + project                     = (known after apply)
      + self_link                   = (known after apply)
      + storage_class               = "STANDARD"
      + uniform_bucket_level_access = true
      + url                         = (known after apply)

      + lifecycle_rule {
          + action {
              + type = "Delete"
            }

          + condition {
              + age                   = 30
              + matches_storage_class = []
              + with_state            = (known after apply)
            }
        }

      + versioning {
          + enabled = true
        }
    }

Plan: 2 to add, 0 to change, 0 to destroy.

Do you want to perform these actions?
  Terraform will perform the actions described above.
  Only 'yes' will be accepted to approve.

  Enter a value: yes

google_bigquery_dataset.dataset: Creating...
google_storage_bucket.data-lake-bucket: Creating...
google_storage_bucket.data-lake-bucket: Creation complete after 2s [id=dtc_data_lake_keen-philosophy-339115]
google_bigquery_dataset.dataset: Creation complete after 2s [id=projects/keen-philosophy-339115/datasets/trips_data_all]

Apply complete! Resources: 2 added, 0 changed, 0 destroyed.
```

## Prepare Postgres

Run Postgres and load data as shown in the videos

We'll use the yellow taxi trips from January 2021:

```bash
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2021-01.csv
```

You will also need the dataset with zones:

```bash 
wget https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv
```

Download this data and put it to Postgres

## Question 3. Count records

How many taxi trips were there on January 15?

Consider only trips that started on January 15.

**Answer:** 53024

**Code:**

```sql
SELECT COUNT(1)
FROM 
	yellow_taxi_data
WHERE 
	CAST(tpep_pickup_datetime AS DATE) = '2021-01-15';
```

## Question 4. Average

Find the largest tip for each day.
On which day it was the largest tip in January?

(note: it's not a typo, it's "tip", not "trip")

**Answer:** 2021-01-20

**Code:**

```sql
SELECT 
	CAST(tpep_pickup_datetime AS DATE) pickup,
	MAX(tip_amount) max_tip
FROM 
	yellow_taxi_data
GROUP BY 
	pickup 
ORDER BY 
	max_tip DESC;
```

## Question 5. Most popular destination

What was the most popular destination for passengers picked up in central park on January 14?

Enter the zone name (not id). If the zone name is unknown (missing), write "Unknown"

**Answer:** Upper East Side South

**Code:**

```sql
SELECT 
	zdo."Zone" AS dropoff_loc,
	COUNT(zdo."Zone") AS n_dropoff_loc
FROM 
	yellow_taxi_data t JOIN zones zpu
	 ON t."PULocationID" = zpu."LocationID"
	JOIN zones zdo
	 ON t."DOLocationID" = zdo."LocationID"
WHERE 
	CAST(tpep_pickup_datetime AS DATE) = '2021-01-14' AND
	zpu."Zone" = 'Central Park'
GROUP BY 
	dropoff_loc
ORDER BY 
	n_dropoff_loc DESC;
```

## Question 6.

What's the pickup-dropoff pair with the largest 
average price for a ride (calculated based on `total_amount`)?

Enter two zone names separated by a slash

For example:

"Jamaica Bay / Clinton East"

If any of the zone names are unknown (missing), write "Unknown". For example, "Unknown / Clinton East".

**Answer:** Alphabet City / Unknown

**Code:**

```sql
SELECT 
	CONCAT(zpu."Zone", ' / ', zdo."Zone") AS pickup_dropoff,
	AVG(t."total_amount") AS avg_pickup_dropoff
FROM 
	yellow_taxi_data t JOIN zones zpu
	 ON t."PULocationID" = zpu."LocationID"
	JOIN zones zdo
	 ON t."DOLocationID" = zdo."LocationID"
GROUP BY 
	pickup_dropoff
ORDER BY 
	avg_pickup_dropoff DESC;
```

## Submitting the solutions

* Form for submitting: https://forms.gle/yGQrkgRdVbiFs8Vd7
* You can submit your homework multiple times. In this case, only the last submission will be used. 

Deadline: 24 January, 17:00 CET
