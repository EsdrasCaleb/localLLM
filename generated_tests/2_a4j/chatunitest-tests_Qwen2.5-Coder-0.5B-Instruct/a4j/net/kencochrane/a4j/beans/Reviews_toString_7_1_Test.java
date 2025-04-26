package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Reviews_toString_7_1_Test {

    @Mock
    private Reviews reviews;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testToString() {
        // Arrange
        when(reviews.getAvgCustomerRating()).thenReturn("4.5");
        when(reviews.getTotalCustomerReviews()).thenReturn("100");
        when(reviews.getReviewsArrayList()).thenReturn(new ArrayList<>());
        // Act
        String result = reviews.toString();
        // Assert
        assertEquals("4.5\n100\n# of reviews = 0", result);
    }
}
