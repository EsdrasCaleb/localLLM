package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Reviews_toString_7_3_Test {

    @Test
    void testToString() {
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("100");
        reviews.setCustomerReview(new CustomerReview[] { new CustomerReview(), new CustomerReview() });
        reviews.getReviewsArrayList();
        String expected = "4.5\n100\nCustomer Review 1\nCustomer Review 2\n# of reviews = 2";
        String actual = reviews.toString();
        assertEquals(expected, actual);
    }
}
