package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class SellerFeedback_toString_3_1_Test {

    @Test
    public void testToString() {
        // Create a SellerFeedback object
        SellerFeedback sellerFeedback = new SellerFeedback();
        sellerFeedback.setFeedback(new FeedBack[] { new FeedBack(), new FeedBack(), new FeedBack() });
        // Call the toString method
        String result = sellerFeedback.toString();
        // Check if the result is not null and contains the expected feedbacks
        assert result != null;
        assert result.contains("feedbacks is null ");
        // Check if the feedbacks array contains the expected feedbacks
        FeedBack[] expectedFeedbacks = { new FeedBack(), new FeedBack(), new FeedBack() };
        assert expectedFeedbacks.length == sellerFeedback.getFeedbackArrayList().size();
        for (int i = 0; i < expectedFeedbacks.length; i++) {
            assert expectedFeedbacks[i] == sellerFeedback.getFeedbackArrayList().get(i);
        }
    }
}
