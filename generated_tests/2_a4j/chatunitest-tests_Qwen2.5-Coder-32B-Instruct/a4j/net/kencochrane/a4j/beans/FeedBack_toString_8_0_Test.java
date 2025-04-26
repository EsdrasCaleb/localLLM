package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FeedBack_toString_8_0_Test {

    private FeedBack feedBack;

    @BeforeEach
    public void setUp() {
        feedBack = new FeedBack();
    }

    @Test
    public void testToString_AllFieldsSet() {
        // Arrange
        feedBack.setFeedbackRater("John Doe");
        feedBack.setFeedbackRating("5");
        feedBack.setFeedbackComments("Excellent service!");
        feedBack.setFeedbackDate("2023-10-01");
        String expectedOutput = "--------------- \n" + "Rater = John Doe\n" + "Rating = 5\n" + "Comments = Excellent service!\n" + "Date = 2023-10-01\n" + "--------------- \n";
        // Act
        String actualOutput = feedBack.toString();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToString_SomeFieldsSet() {
        // Arrange
        feedBack.setFeedbackRater("Jane Smith");
        feedBack.setFeedbackRating("3");
        String expectedOutput = "--------------- \n" + "Rater = Jane Smith\n" + "Rating = 3\n" + "Comments = null\n" + "Date = null\n" + "--------------- \n";
        // Act
        String actualOutput = feedBack.toString();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToString_NoFieldsSet() {
        // Arrange
        String expectedOutput = "--------------- \n" + "Rater = null\n" + "Rating = null\n" + "Comments = null\n" + "Date = null\n" + "--------------- \n";
        // Act
        String actualOutput = feedBack.toString();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
