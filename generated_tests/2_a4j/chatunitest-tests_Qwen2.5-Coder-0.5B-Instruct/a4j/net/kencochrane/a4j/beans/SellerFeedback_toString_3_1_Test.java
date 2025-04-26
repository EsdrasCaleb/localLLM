package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class SellerFeedback_toString_3_1_Test {

    private SellerFeedback sellerFeedback;

    @BeforeEach
    public void setUp() {
        sellerFeedback = new SellerFeedback();
        sellerFeedback.setFeedback(new FeedBack[] { new FeedBack(), new FeedBack() });
    }

    @Test
    public void testToString() {
        assertEquals("feedbacks is null\n# of feedbacks = 2", sellerFeedback.toString());
    }
}
