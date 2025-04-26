package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Reviews_toString_7_1_Test {

    @Test
    void testToString_withReviews() {
        CustomerReview review1 = new CustomerReview("Excellent", 5);
        CustomerReview review2 = new CustomerReview("Good", 4);
        CustomerReview[] reviewsArray = { review1, review2 };
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("100");
        reviews.setCustomerReview(reviewsArray);
        String expectedOutput = "4.5\n100\nExcellent 5\nGood 4\n# of reviews = 2";
        String actualOutput = reviews.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testToString_withNullReviews() {
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("100");
        String expectedOutput = "4.5\n100\nreviews is null ";
        String actualOutput = reviews.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testToString_withEmptyReviews() {
        CustomerReview[] reviewsArray = {};
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("0");
        reviews.setCustomerReview(reviewsArray);
        String expectedOutput = "4.5\n0\nreviews is null ";
        String actualOutput = reviews.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    // Additional test case for edge cases
    @Test
    void testToString_withNullAvgCustomerRating() {
        CustomerReview review1 = new CustomerReview("Excellent", 5);
        CustomerReview[] reviewsArray = { review1 };
        Reviews reviews = new Reviews();
        reviews.setTotalCustomerReviews("100");
        reviews.setCustomerReview(reviewsArray);
        String expectedOutput = "null\n100\nExcellent 5\n# of reviews = 1";
        String actualOutput = reviews.toString();
        // Explicitly check for null
        assertNull(reviews.getAvgCustomerRating());
        assertEquals(expectedOutput, actualOutput);
    }
}

// CustomerReview class (needed for the test)
class CustomerReview {

    String feedback;

    int rating;

    public CustomerReview(String feedback, int rating) {
        this.feedback = feedback;
        this.rating = rating;
    }

    public CustomerReview() {
        // Default constructor
    }

    @Override
    public String toString() {
        return feedback + " " + rating;
    }
}
