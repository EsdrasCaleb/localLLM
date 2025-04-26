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

public class CustomerReview_toString_6_0_Test {

    @Test
    public void testToString() {
        CustomerReview review = new CustomerReview();
        review.setRating("5");
        review.setSummary("Excellent service!");
        review.setComment("This is a great product. I love it!");
        String expectedOutput = "5\nExcellent service!\nThis is a great product. I love it!";
        String actualOutput = review.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
