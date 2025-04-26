package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class SellerFeedback_toString_3_4_Test {

    @Mock
    private SellerFeedback sellerFeedback;

    @BeforeEach
    public void setUp() {
        // <Buggy Line>: constructor FeedBack in class net.kencochrane.a4j.beans.FeedBack cannot be applied to given types;  required: no arguments  found:    java.lang.String  reason: actual and formal argument lists differ in length
        ArrayList<FeedBack> feedbacks = new ArrayList<>();
        feedbacks.add(new FeedBack());
        feedbacks.add(new FeedBack());
        when(sellerFeedback.getFeedbackArrayList()).thenReturn(feedbacks);
    }

    @Test
    public void testToString() {
        String expected = "Feedback 1\nFeedback 2\n# of feedbacks = 2";
        assertEquals(expected, sellerFeedback.toString());
    }
}
