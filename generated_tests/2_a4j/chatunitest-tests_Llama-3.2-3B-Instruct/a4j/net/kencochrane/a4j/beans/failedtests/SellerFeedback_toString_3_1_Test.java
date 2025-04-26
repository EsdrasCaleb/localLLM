// Test method
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
    public void testToString_SimpleCase() {
        SellerFeedback sellerFeedback = new SellerFeedback();
        FeedBack feedback1 = Mockito.mock(FeedBack.class);
        FeedBack feedback2 = Mockito.mock(FeedBack.class);
        when(feedback1.toString()).thenReturn("This is a great product!");
        when(feedback2.toString()).thenReturn("I love this product!");
        sellerFeedback.addFeedback(feedback1);
        sellerFeedback.addFeedback(feedback2);
        String expectedOutput = "This is a great product!\nI love this product!\n# of feedbacks = 2";
        assertEquals(expectedOutput, sellerFeedback.toString());
    }

    @Test
    public void testToString_NullFeedbacks() {
        SellerFeedback sellerFeedback = new SellerFeedback();
        String expectedOutput = "feedbacks is null";
        assertEquals(expectedOutput, sellerFeedback.toString());
    }

    @Test
    public void testToString_EmptyFeedbacks() {
        SellerFeedback sellerFeedback = new SellerFeedback();
        String expectedOutput = "";
        assertEquals(expectedOutput, sellerFeedback.toString());
    }

    @Test
    public void testToString_NullSellerFeedback() {
        SellerFeedback sellerFeedback = null;
        assertThrows(NullPointerException.class, () -> sellerFeedback.toString());
    }
}

class SellerFeedback implements Serializable {

    private ArrayList<FeedBack> feedbacks = new ArrayList<>();

    public SellerFeedback() {
    }

    public void addFeedback(FeedBack feedback) {
        feedbacks.add(feedback);
    }

    @Override
    public String toString() {
        StringBuffer output = new StringBuffer();
        for (FeedBack feed : feedbacks) {
            output.append(feed.toString() + "\n");
        }
        if (feedbacks.isEmpty()) {
            output.append("feedbacks is empty");
        } else {
            output.append("# of feedbacks = ").append(feedbacks.size());
        }
        return output.toString();
    }
}

class FeedBack implements Serializable {

    private String message;

    public FeedBack(String message) {
        this.message = message;
    }

    @Override
    public String toString() {
        return message;
    }
}

class FeedBackMock {

    private String message;

    public FeedBackMock(String message) {
        this.message = message;
    }

    @Override
    public String toString() {
        return message;
    }
}
