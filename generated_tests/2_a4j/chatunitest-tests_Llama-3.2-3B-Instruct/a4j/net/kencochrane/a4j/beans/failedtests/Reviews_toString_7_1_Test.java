package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Reviews_toString_7_1_Test {

    private Reviews reviews;

    @BeforeEach
    public void setup() {
        reviews = new Reviews();
    }

    @Test
    public void testToString() {
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("10");
        List<CustomerReview> reviewsList = new ArrayList<>();
        reviewsList.add(new CustomerReview());
        reviewsList.add(new CustomerReview());
        reviews.setCustomerReview((CustomerReview[]) reviewsList.toArray(new CustomerReview[0]));
        assertNotNull(reviews.toString());
    }

    @Test
    public void testToString_EmptyReviews() {
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("10");
        List<CustomerReview> reviewsList = new ArrayList<>();
        reviews.setCustomerReview((CustomerReview[]) reviewsList.toArray(new CustomerReview[0]));
        assertEquals("4.5\n10\nreviews is null ", reviews.toString());
    }

    @Test
    public void testToString_NullReviews() {
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("10");
        reviews.setCustomerReview(null);
        assertEquals("4.5\n10\nreviews is null ", reviews.toString());
    }
}
