package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_31_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingAbove_withValidRating() {
        String rating = "AAA";
        scannerSubscription.moodyRatingAbove(rating);
        // Using reflection to access the private field
        String actualRating = getPrivateField(scannerSubscription, "m_moodyRatingAbove");
        assertEquals(rating, actualRating);
    }

    @Test
    public void testMoodyRatingAbove_withEmptyString() {
        String rating = "";
        scannerSubscription.moodyRatingAbove(rating);
        // Using reflection to access the private field
        String actualRating = getPrivateField(scannerSubscription, "m_moodyRatingAbove");
        assertEquals(rating, actualRating);
    }

    @Test
    public void testMoodyRatingAbove_withNull() {
        String rating = null;
        scannerSubscription.moodyRatingAbove(rating);
        // Using reflection to access the private field
        String actualRating = getPrivateField(scannerSubscription, "m_moodyRatingAbove");
        assertEquals(rating, actualRating);
    }

    @SuppressWarnings("unchecked")
    private <T> T getPrivateField(Object obj, String fieldName) {
        try {
            java.lang.reflect.Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return (T) field.get(obj);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
