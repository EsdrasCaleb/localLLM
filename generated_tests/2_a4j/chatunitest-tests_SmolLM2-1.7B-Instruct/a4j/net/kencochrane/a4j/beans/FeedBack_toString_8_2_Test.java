package net.kencochrane.a4j.beans;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

@RunWith(MockitoJUnitRunner.class)
public class FeedBack_toString_8_2_Test {

    @Mock
    private FeedBack mockFeedBack;

    @InjectMocks
    private FeedBack testFeedBack;

    @Test
    public void testToString() {
        // Arrange
        testFeedBack.setFeedbackRating("5");
        testFeedBack.setFeedbackComments("This is a great feedback!");
        testFeedBack.setFeedbackDate("2022-01-01");
        testFeedBack.setFeedbackRater("John Doe");
        // Act
        String result = testFeedBack.toString();
        // Assert
        assertNotNull(result);
        assertEquals("Rater = John Doe\nRating = 5\nComments = This is a great feedback!\nDate = 2022-01-01\n---------------", result);
    }
}
