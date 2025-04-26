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
    void testToString() {
        CustomerReview customerReview = new CustomerReview();
        customerReview.setRating("5");
        customerReview.setSummary("This is a great product.");
        customerReview.setComment("This product is very good.");
        String expectedString = "5\nThis is a great product.\nThis product is very good.";
        String actualString = customerReview.toString();
        assertEquals(expectedString, actualString);
    }
}
