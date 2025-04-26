package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Reviews_toString_7_0_Test {

    @Test
    public void testToString() throws Exception {
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("100");
        ArrayList<CustomerReview> reviewsList = new ArrayList<>();
        reviewsList.add(new CustomerReview());
        reviews.setCustomerReview(reviewsList.toArray(new CustomerReview[0]));
        Field field = Reviews.class.getDeclaredField("reviews");
        field.setAccessible(true);
        field.set(reviews, reviewsList);
        String expected = "4.5\n100\n# of reviews = 1\n";
        assertEquals(expected, reviews.toString());
    }
}
