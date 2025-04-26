package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Reviews_toString_7_0_Test {

    private Reviews reviews;

    @BeforeEach
    public void setUp() {
        reviews = new Reviews();
    }

    @Test
    public void testToString() {
        CustomerReview mockReview = mock(CustomerReview.class);
        ArrayList<CustomerReview> reviewList = new ArrayList<>();
        reviewList.add(mockReview);
        reviews.setCustomerReview(reviewList.toArray(new CustomerReview[0]));
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("10");
        when(mockReview.toString()).thenReturn("Mock Customer Review");
        String expected = "4.5\n10\nMock Customer Review\n# of reviews = 1";
        assertEquals(expected, reviews.toString());
    }
}
