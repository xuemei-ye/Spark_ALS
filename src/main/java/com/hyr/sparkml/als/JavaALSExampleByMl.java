package com.hyr.sparkml.als;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
//import org.apache.spark;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.lang3.math.NumberUtils;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.function.Function;
// Import factory methods provided by DataTypes.
import org.apache.spark.sql.types.DataTypes;
// Import StructType and StructField
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
// Import Row.
import org.apache.spark.sql.Row;
// Import RowFactory.
import org.apache.spark.sql.RowFactory;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;

/**
 * @author huangyueran
 * @category ALS-WR
 */


public class JavaALSExampleByMl {

    private static final Logger log = LoggerFactory.getLogger(JavaALSExampleByMl.class);

    public static class Rating implements Serializable {
        // 0::2::3::1424380312
        private int userId; // 0
        private int movieId; // 2
        private float rating; // 3
//        private long timestamp; // 1424380312

        public Rating() {
        }

        public Rating(int userId, int movieId, float rating) {
            this.userId = userId;
            this.movieId = movieId;
            this.rating = rating;
//            this.timestamp = timestamp;
        }

        public int getUserId() {
            return userId;
        }

        public int getMovieId() {
            return movieId;
        }

        public float getRating() {
            return rating;
        }

//        public long getTimestamp() {
//            return timestamp;
//        }

        public static Rating parseRating(String str) {
            String[] fields = str.split("\001");
            int i = 0;
            if(str.equals("null")){
                return new Rating(-1,-1,Float.parseFloat("-1"));
            }
            // 将非法字符、空字符 过滤
            for(i=0;i<fields.length;i++){
                if(fields[i].equals("null") || !NumberUtils.isDigits(fields[0]) || !NumberUtils.isDigits(fields[1])){
                    return new Rating(-1,-1,Float.parseFloat("-1"));
                }
            }
            if (fields.length != 3) {
                throw new IllegalArgumentException("Each line must contain 3 fields");
            }
            int userId = Integer.parseInt(fields[0]);

            int len_vedio_id = fields[1].split("_").length;
            int movieId = Integer.parseInt(fields[1].split("_")[len_vedio_id-1]);
            float rating = Float.parseFloat(fields[2]);

            return new Rating(userId, movieId, rating);
        }
    }

    public static void main(String[] args) {
        String model_path = "/user/meiyingt/als.model";
        // /hiveweb/tempdb.db/user_video_score

//        SparkSession spark = SparkSession.builder().getOrCreate();

        SparkConf conf = new SparkConf().setAppName("JavaALSExample");//.setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(jsc);

        JavaRDD<Rating> ratingsRDD = jsc.textFile("/user/meiyingt/tmp")
                .map(new Function<String, Rating>() {
                    public Rating call(String str) {
                        return Rating.parseRating(str);
                    }
                });

        sqlContext. Dataset<Rating> ratingsRDD.

        Dataset<Row> ratings = sqlContext.createDataFrame(ratingsRDD,Rating.class).as();

//        Dataset<Row> ratings = sqlContext.createDataFrame(ratingsRDD, Rating.class);
        Dataset<Row>[] splits = ratings.randomSplit(new double[]{0.8, 0.2}); // //对数据进行分割，80%为训练样例，剩下的为测试样例。
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        // Build the recommendation model using ALS on the training data
        ALS als = new ALS().setMaxIter(1) // 设置迭代次数
                .setRegParam(0.01) // //正则化参数，使每次迭代平滑一些，此数据集取0.1好像错误率低一些。
                .setRank(20)
                .setUserCol("userId").setItemCol("movieId")
                .setRatingCol("rating");
        ALSModel model = als.fit(training); // //调用算法开始训练

        model.write().overwrite().saveImpl(model_path);

        Dataset<Row> itemFactors = model.itemFactors();
        itemFactors.show(1500);
        Dataset<Row> userFactors = model.userFactors();
        userFactors.show();

        // Evaluate the model by computing the RMSE on the test data
        Dataset<Row> rawPredictions = model.transform(test); //对测试数据进行预测
        Dataset<Row> predictions = rawPredictions
                .withColumn("rating", rawPredictions.col("rating").cast(DataTypes.DoubleType))
                .withColumn("prediction", rawPredictions.col("prediction").cast(DataTypes.DoubleType));

        RegressionEvaluator evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating")
                .setPredictionCol("prediction");
        Double rmse = evaluator.evaluate(predictions);
        log.info("Root-mean-square error = {} ", rmse);

        System.out.print("Success !!!");
        jsc.stop();
    }
}
