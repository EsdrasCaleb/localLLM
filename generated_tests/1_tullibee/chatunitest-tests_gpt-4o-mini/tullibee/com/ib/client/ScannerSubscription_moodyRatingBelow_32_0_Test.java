package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_32_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingBelow() {
        // Test with a valid Moody's rating
        String rating = "Baa3";
        scannerSubscription.moodyRatingBelow(rating);
        // Use reflection to access the private field m_moodyRatingBelow
        String moodyRatingBelow = getPrivateField(scannerSubscription, "m_moodyRatingBelow");
        assertEquals(rating, moodyRatingBelow);
        // Test with null
        scannerSubscription.moodyRatingBelow(null);
        moodyRatingBelow = getPrivateField(scannerSubscription, "m_moodyRatingBelow");
        assertEquals(null, moodyRatingBelow);
        // Test with an empty string
        scannerSubscription.moodyRatingBelow("");
        moodyRatingBelow = getPrivateField(scannerSubscription, "m_moodyRatingBelow");
        assertEquals("", moodyRatingBelow);
    }

    private String getPrivateField(ScannerSubscription scannerSubscription, String fieldName) {
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (String) field.get(scannerSubscription);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
