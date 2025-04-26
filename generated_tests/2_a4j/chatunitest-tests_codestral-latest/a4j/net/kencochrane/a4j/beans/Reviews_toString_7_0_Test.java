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

    @Mock
    private ArrayList<CustomerReview> reviews;

    @InjectMocks
    private Reviews review;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        review.setAvgCustomerRating("4.5");
        review.setTotalCustomerReviews("100");
    }

    @Test
    void testToStringWithReviews() {
        CustomerReview mockReview1 = mock(CustomerReview.class);
        CustomerReview mockReview2 = mock(CustomerReview.class);
        when(mockReview1.toString()).thenReturn("Review 1");
        when(mockReview2.toString()).thenReturn("Review 2");
        when(reviews.size()).thenReturn(2);
        when(reviews.get(0)).thenReturn(mockReview1);
        when(reviews.get(1)).thenReturn(mockReview2);
        review.setCustomerReview(new CustomerReview[] { mockReview1, mockReview2 });
        String expected = "4.5\n100\nReview 1\nReview 2\n# of reviews = 2";
        assertEquals(expected, review.toString());
    }
}
