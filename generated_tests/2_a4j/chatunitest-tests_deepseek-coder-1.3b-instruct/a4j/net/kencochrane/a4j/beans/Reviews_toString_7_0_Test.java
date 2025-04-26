package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Reviews_toString_7_0_Test {

    @Test
    void testToString() {
        Reviews review = new Reviews();
        review.setAvgCustomerRating("4.5");
        review.setTotalCustomerReviews("100");
        CustomerReview customerReview = new CustomerReview();
        review.setCustomerReview(new CustomerReview[] { customerReview });
        String expected = "4.5\n100\n" + customerReview + "\n# of reviews = 1";
        assertEquals(expected, review.toString());
    }
}
