package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class SellerFeedback_toString_3_3_Test {

    @Test
    public void testToString_FeedbacksAreNotNull() {
        // Arrange
        SellerFeedback sellerFeedback = new SellerFeedback();
        FeedBack feed1 = Mockito.mock(FeedBack.class);
        FeedBack feed2 = Mockito.mock(FeedBack.class);
        FeedBack feed3 = Mockito.mock(FeedBack.class);
        // Act
        sellerFeedback.setFeedback(new FeedBack[] { feed1, feed2, feed3 });
        String output = sellerFeedback.toString();
        // Assert
        assertEquals(output, "feed1\nfeed2\nfeed3\n# of feedbacks = 3");
    }

    @Test
    public void testToString_FeedbacksAreNull() {
        // Arrange
        SellerFeedback sellerFeedback = new SellerFeedback();
        // Act and Assert
        assertTrue(sellerFeedback.toString().contains("feedbacks is null"));
    }
}
