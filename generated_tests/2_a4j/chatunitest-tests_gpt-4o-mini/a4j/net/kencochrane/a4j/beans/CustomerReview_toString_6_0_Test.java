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

    private CustomerReview customerReview;

    @BeforeEach
    public void setUp() {
        customerReview = new CustomerReview();
    }

    @Test
    public void testToString_withNullValues() {
        // Given
        customerReview.setRating(null);
        customerReview.setSummary(null);
        customerReview.setComment(null);
        // When
        String result = customerReview.toString();
        // Then
        assertEquals("\n\n\n", result);
    }

    @Test
    public void testToString_withEmptyValues() {
        // Given
        customerReview.setRating("");
        customerReview.setSummary("");
        customerReview.setComment("");
        // When
        String result = customerReview.toString();
        // Then
        assertEquals("\n\n\n", result);
    }

    @Test
    public void testToString_withNonEmptyValues() {
        // Given
        customerReview.setRating("5 stars");
        customerReview.setSummary("Great product");
        customerReview.setComment("I love it!");
        // When
        String result = customerReview.toString();
        // Then
        assertEquals("5 stars\nGreat product\nI love it!\n", result);
    }

    @Test
    public void testToString_withHTMLComment() {
        // Given
        customerReview.setRating("4 stars");
        customerReview.setSummary("Good product");
        customerReview.setComment("<b>Highly recommend!</b>");
        // When
        String result = customerReview.toString();
        // Then
        assertEquals("4 stars\nGood product\nHighly recommend!\n", result);
    }
}
