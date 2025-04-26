package com.ib.client;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.ib.client.ScannerSubscription;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class ScannerSubscription_moodyRatingBelow_32_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    void testMoodyRatingBelow_ValidInput() {
        String rating = "Aaa";
        scannerSubscription.moodyRatingBelow(rating);
        assertEquals(rating, getMoodyRatingBelow(scannerSubscription));
    }

    @Test
    void testMoodyRatingBelow_NullInput() {
        String rating = null;
        scannerSubscription.moodyRatingBelow(rating);
        assertNull(getMoodyRatingBelow(scannerSubscription));
    }

    @Test
    void testMoodyRatingBelow_EmptyInput() {
        String rating = "";
        scannerSubscription.moodyRatingBelow(rating);
        assertEquals("", getMoodyRatingBelow(scannerSubscription));
    }

    @Test
    void testMoodyRatingBelow_InputWithSpaces() {
        String rating = "   Aaa   ";
        scannerSubscription.moodyRatingBelow(rating);
        assertEquals(rating.trim(), getMoodyRatingBelow(scannerSubscription));
    }

    @Test
    void testMoodyRatingBelow_ExistingValue() {
        String existingRating = "Baa3";
        scannerSubscription.moodyRatingBelow(existingRating);
        String newRating = "Aaa";
        scannerSubscription.moodyRatingBelow(newRating);
        assertEquals(newRating, getMoodyRatingBelow(scannerSubscription));
    }

    private String getMoodyRatingBelow(ScannerSubscription subscription) {
        try {
            Field field = ScannerSubscription.class.getDeclaredField("m_moodyRatingBelow");
            field.setAccessible(true);
            return (String) field.get(subscription);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Error accessing private field.");
            // This line is important for compilation
            return null;
        }
    }
}
