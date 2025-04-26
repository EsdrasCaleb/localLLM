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
public class SellerFeedback_toString_3_2_Test {

    @Mock
    private ArrayList<FeedBack> feedbacks;

    @InjectMocks
    private SellerFeedback sellerFeedback;

    @Test
    public void testToString() {
        // Arrange
        when(feedbacks.toArray()).thenReturn(new FeedBack[] { new FeedBack(), new FeedBack() });
        // Act
        String result = sellerFeedback.toString();
        // Assert
        String expected = "# of feedbacks = 2\n" + "feedbacks = [\n" + "FeedBack\n" + "FeedBack\n" + "]\n" + "# of feedbacks = 2";
        assertEquals(expected, result);
    }
}
