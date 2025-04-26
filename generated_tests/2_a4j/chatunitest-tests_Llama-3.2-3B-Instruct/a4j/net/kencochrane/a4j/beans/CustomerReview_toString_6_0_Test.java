package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Arrays;
import java.util.regex.Pattern;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class CustomerReview_toString_6_0_Test {

    @Test
    public void testToString() {
        // Arrange
        CustomerReview review = new CustomerReview();
        review.setRating("5");
        review.setSummary("This is a review");
        review.setComment("<p>This is a <b>test</b> review</p>");
        // Act
        String result = review.toString();
        // Assert
        assertEquals("5\nThis is a review\n<p>This is a <b>test</b> review</p>", result);
        // Test with null comment
        review.setComment(null);
        result = review.toString();
        assertEquals("5\nThis is a review\n", result);
        // Test with comment containing HTML tags
        review.setComment("<p>This is a <b>test</b> review</p>");
        result = review.toString();
        assertEquals("5\nThis is a review\n<p>This is a <b>test</b> review</p>", result);
        // Test with comment containing newline characters
        review.setComment("This is a review\nwith\nnewline characters");
        result = review.toString();
        assertEquals("5\nThis is a review\nwith\nnewline characters", result);
        // Test with invalid comment
        review.setComment("This is a review");
        result = review.toString();
        assertEquals("5\nThis is a review\n", result);
    }

    @Test
    public void testToString_ThrowsException() {
        // Arrange
        CustomerReview review = new CustomerReview();
        // Act and Assert
        assertThrows(NullPointerException.class, () -> review.toString());
    }
}
