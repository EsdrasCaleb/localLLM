package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerFeedback_toString_3_0_Test {

    @InjectMocks
    private SellerFeedback sellerFeedback;

    @Mock
    private ArrayList<FeedBack> mockFeedbacks;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToStringWithFeedbacks() {
        FeedBack feedback1 = new FeedBack();
        FeedBack feedback2 = new FeedBack();
        when(mockFeedbacks.size()).thenReturn(2);
        when(mockFeedbacks.get(0)).thenReturn(feedback1);
        when(mockFeedbacks.get(1)).thenReturn(feedback2);
        sellerFeedback.setFeedback(new FeedBack[] { feedback1, feedback2 });
        String expected = feedback1.toString() + "\n" + feedback2.toString() + "\n" + "# of feedbacks = 2";
        String actual = sellerFeedback.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringWithNoFeedbacks() {
        when(mockFeedbacks.size()).thenReturn(0);
        sellerFeedback.setFeedback(new FeedBack[] {});
        String expected = "# of feedbacks = 0";
        String actual = sellerFeedback.toString();
        assertEquals(expected, actual);
    }
}
