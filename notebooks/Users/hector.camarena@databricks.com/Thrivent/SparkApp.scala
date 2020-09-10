// Databricks notebook source
//Testing Change

val diamonds = spark.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv")

display(diamonds)

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._



val transform = (diamonds.groupBy("cut")
                  .agg(
                    avg("price").cast(IntegerType).as("avg_price"),
                    avg("x").cast(IntegerType).as("avg_x"),
                    avg("y").cast(IntegerType).as("avg_y"),
                    avg("z").cast(IntegerType).as("avg_z"),
                    max("carat").as("max_carat"))
                )

display(transform)

// COMMAND ----------

transform.write.mode("append").format("delta").saveAsTable("camarena.diamonds_enriched_delta")

// COMMAND ----------

