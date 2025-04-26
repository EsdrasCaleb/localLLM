package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Reviews_toString_7_2_Test {

    @Test
    void testToString_noReviews() {
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("3.0");
        reviews.setTotalCustomerReviews("5");
        String expected = "3.0\n5\nreviews is null ";
        assertEquals(expected, reviews.toString());
    }

    @Test
    void testToString_nullAvgRating() {
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating(null);
        reviews.setTotalCustomerReviews("5");
        String expected = "null\n5\nreviews is null ";
        assertEquals(expected, reviews.toString());
    }

    @Test
    void testToString_nullTotalReviews() {
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("4.0");
        reviews.setTotalCustomerReviews(null);
        String expected = "4.0\nnull\nreviews is null ";
        assertEquals(expected, reviews.toString());
    }

    class CustomerReview {

        private String customerName;

        private String rating;

        private String reviewText;

        public String getCustomerName() {
            return customerName;
        }

        public void setCustomerName(String customerName) {
            this.customerName = customerName;
        }

        public String getRating() {
            return rating;
        }

        public void setRating(String rating) {
            this.rating = rating;
        }

        public String getReviewText() {
            return reviewText;
        }

        public void setReviewText(String reviewText) {
            this.reviewText = reviewText;
        }

        @Override
        public String toString() {
            return "Customer Name: " + customerName + ", Rating: " + rating + ", Review: " + reviewText;
        }
    }
}
