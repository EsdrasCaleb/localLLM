package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FeedBack_toString_8_0_Test {

    private FeedBack feedBack;

    @BeforeEach
    public void setUp() {
        feedBack = mock(FeedBack.class);
    }

    @Test
    public void testToString() {
        when(feedBack.getFeedbackRating()).thenReturn("Excellent");
        when(feedBack.getFeedbackComments()).thenReturn("Great job!");
        when(feedBack.getFeedbackDate()).thenReturn("2023-10-01");
        when(feedBack.getFeedbackRater()).thenReturn("John Doe");
        String expectedOutput = "--------------- \n";
        expectedOutput += "Rater = John Doe\n";
        expectedOutput += "Rating = Excellent\n";
        expectedOutput += "Comments = Great job!\n";
        expectedOutput += "Date = 2023-10-01\n";
        expectedOutput += "--------------- \n";
        String actualOutput = feedBack.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
