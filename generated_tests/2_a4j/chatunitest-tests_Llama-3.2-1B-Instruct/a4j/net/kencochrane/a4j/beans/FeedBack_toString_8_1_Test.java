package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class FeedBack_toString_8_1_Test {

    @Mock
    private FeedBack feedBack;

    @InjectMocks
    private FeedBack testFeedBack;

    @Test
    public void testToString() {
        // Arrange
        when(feedBack.getFeedbackRating()).thenReturn("5");
        when(feedBack.getFeedbackComments()).thenReturn("Comments");
        when(feedBack.getFeedbackDate()).thenReturn("2022-01-01");
        when(feedBack.getFeedbackRater()).thenReturn("John");
        // Act
        String output = testFeedBack.toString();
        // Assert
        String expectedOutput = "--------------- \nRater = John \nRating = 5 \nComments = Comments \nDate = 2022-01-01 \n--------------- \n";
        assertEquals(expectedOutput, output);
    }
}
