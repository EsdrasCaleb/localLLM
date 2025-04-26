package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Reviews_toString_7_1_Test {

    @Test
    public void testToString() {
        Reviews reviews = new Reviews();
        reviews.setAvgCustomerRating("4.5");
        reviews.setTotalCustomerReviews("100");
        String expectedOutput = "4.5\n100\n# of reviews = 0";
        assertEquals(expectedOutput, reviews.toString());
    }
}
