package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

public class CustomerReview_toString_6_1_Test {

    @Test
    public void testToString() {
        // Arrange
        String expectedOutput = "<Rating>\n<Summary>\n<Comment>\n";
        CustomerReview customerReview = new CustomerReview();
        customerReview.setRating("<Rating>");
        customerReview.setSummary("<Summary>");
        customerReview.setComment("<Comment>");
        // Act
        String actualOutput = customerReview.toString();
        // Assert
        Assertions.assertEquals(expectedOutput, actualOutput);
    }
}
