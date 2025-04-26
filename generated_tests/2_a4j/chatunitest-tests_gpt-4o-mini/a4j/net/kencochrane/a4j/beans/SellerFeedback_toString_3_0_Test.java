package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

public class SellerFeedback_toString_3_0_Test {

    private SellerFeedback sellerFeedback;

    @BeforeEach
    public void setUp() {
        sellerFeedback = new SellerFeedback();
    }

    @Test
    public void testToString_withFeedbacks() throws Exception {
        // Arrange
        FeedBack feedback1 = new FeedBack();
        feedback1.setFeedbackComments("Great service!");
        FeedBack feedback2 = new FeedBack();
        feedback2.setFeedbackComments("Fast delivery!");
        FeedBack[] feedbacks = { feedback1, feedback2 };
        sellerFeedback.setFeedback(feedbacks);
        // Act
        String result = sellerFeedback.toString();
        // Assert
        assertEquals("Great service!\nFast delivery!\n# of feedbacks = 2", result);
    }

    @Test
    public void testToString_withNoFeedbacks() throws Exception {
        // Arrange
        sellerFeedback.setFeedback(new FeedBack[0]);
        // Act
        String result = sellerFeedback.toString();
        // Assert
        assertEquals("# of feedbacks = 0", result);
    }

    @Test
    public void testToString_withNullFeedbacks() throws Exception {
        // Arrange
        setPrivateField(sellerFeedback, "feedbacks", null);
        // Act
        String result = sellerFeedback.toString();
        // Assert
        assertEquals("feedbacks is null ", result);
    }

    private void setPrivateField(SellerFeedback sellerFeedback, String fieldName, Object value) throws Exception {
        Field field = SellerFeedback.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(sellerFeedback, value);
    }
}
