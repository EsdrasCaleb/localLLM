package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

public class CustomerReview_toString_6_0_Test {

    @Test
    public void testToString() {
        // Arrange
        CustomerReview review = new CustomerReview();
        review.setRating("5");
        review.setSummary("Review summary");
        review.setComment("This is a review comment");
        // Act
        String output = review.toString();
        // Assert
        assertEquals("5\nReview summary\nThis is a review comment\n", output);
    }
}
