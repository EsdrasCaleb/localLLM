package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Reviews_toString_7_0_Test {

    private Reviews reviews;

    private CustomerReview mockReview1;

    private CustomerReview mockReview2;

    @BeforeEach
    void setUp() {
        reviews = new Reviews();
        mockReview1 = mock(CustomerReview.class);
        mockReview2 = mock(CustomerReview.class);
    }

    @Test
    void testToString_WithReviews() {
        // Arrange
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("10");
        ArrayList<CustomerReview> reviewList = new ArrayList<>();
        reviewList.add(mockReview1);
        reviewList.add(mockReview2);
        reviews.setCustomerReview(reviewList.toArray(new CustomerReview[0]));
        when(mockReview1.toString()).thenReturn("Review 1");
        when(mockReview2.toString()).thenReturn("Review 2");
        // Act
        String result = reviews.toString();
        // Assert
        String expected = "4.5\n10\nReview 1\nReview 2\n# of reviews = 2";
        assertEquals(expected, result);
    }

    @Test
    void testToString_NoReviews() {
        // Arrange
        reviews.setAvgCustomerRating("3.0");
        reviews.setTotalCustomerReviews("0");
        reviews.setCustomerReview(new CustomerReview[0]);
        // Act
        String result = reviews.toString();
        // Assert
        String expected = "3.0\n0\n# of reviews = 0";
        assertEquals(expected, result);
    }
}
