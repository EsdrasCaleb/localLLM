package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_33_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testSpRatingAbove() {
        // Test with a valid S&P rating
        String rating = "AA";
        scannerSubscription.spRatingAbove(rating);
        assertEquals(rating, getPrivateField("m_spRatingAbove"));
        // Test with another valid S&P rating
        rating = "BBB";
        scannerSubscription.spRatingAbove(rating);
        assertEquals(rating, getPrivateField("m_spRatingAbove"));
        // Test with null input
        rating = null;
        scannerSubscription.spRatingAbove(rating);
        assertEquals(rating, getPrivateField("m_spRatingAbove"));
        // Test with an empty string
        rating = "";
        scannerSubscription.spRatingAbove(rating);
        assertEquals(rating, getPrivateField("m_spRatingAbove"));
    }

    private String getPrivateField(String fieldName) {
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (String) field.get(scannerSubscription);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
