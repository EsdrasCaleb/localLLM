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

class Reviews_toString_7_0_Test {

    @Mock
    private CustomerReview mockCustomerReview1;

    @Mock
    private CustomerReview mockCustomerReview2;

    private Reviews reviews;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        reviews = new Reviews();
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("100");
    }

    @Test
    void testToStringWithReviews() throws Exception {
        ArrayList<CustomerReview> reviewsList = new ArrayList<>();
        reviewsList.add(mockCustomerReview1);
        reviewsList.add(mockCustomerReview2);
        Field reviewsField = Reviews.class.getDeclaredField("reviews");
        reviewsField.setAccessible(true);
        reviewsField.set(reviews, reviewsList);
        when(mockCustomerReview1.toString()).thenReturn("Review 1 Details");
        when(mockCustomerReview2.toString()).thenReturn("Review 2 Details");
        String expectedOutput = "4.5\n100\nReview 1 Details\nReview 2 Details\n# of reviews = 2";
        assertEquals(expectedOutput, reviews.toString());
    }

    @Test
    void testToStringWithoutReviews() throws Exception {
        Field reviewsField = Reviews.class.getDeclaredField("reviews");
        reviewsField.setAccessible(true);
        reviewsField.set(reviews, null);
        String expectedOutput = "4.5\n100\nreviews is null ";
        assertEquals(expectedOutput, reviews.toString());
    }
}
