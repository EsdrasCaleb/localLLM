package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Reviews_toString_7_0_Test {

    @Mock
    private CustomerReview customerReview;

    @InjectMocks
    private Reviews focal;

    @Test
    public void testToString() {
        // Arrange
        List<CustomerReview> reviews = new ArrayList<>();
        reviews.add(new CustomerReview());
        // Act
        String output = focal.toString();
        // Assert
        String expectedOutput = "CustomerReview{avgCustomerRating=null, reviews=[CustomerReview@...], TotalCustomerReviews=null, # of reviews=0}";
        assertEquals(expectedOutput, output);
    }
}
