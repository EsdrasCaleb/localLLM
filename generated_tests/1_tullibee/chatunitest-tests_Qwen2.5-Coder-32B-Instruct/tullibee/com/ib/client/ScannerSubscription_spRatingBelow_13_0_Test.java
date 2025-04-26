package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingBelow_13_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testSpRatingBelow_DefaultValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        Field spRatingBelowField = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
        spRatingBelowField.setAccessible(true);
        // Set to default value (null)
        spRatingBelowField.set(scannerSubscription, null);
        // When
        String result = scannerSubscription.spRatingBelow();
        // Then
        assertNull(result, "The default value of m_spRatingBelow should be null");
    }

    @Test
    public void testSpRatingBelow_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        String testRating = "BBB";
        Field spRatingBelowField = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
        spRatingBelowField.setAccessible(true);
        spRatingBelowField.set(scannerSubscription, testRating);
        // When
        String result = scannerSubscription.spRatingBelow();
        // Then
        assertEquals(testRating, result, "The value of m_spRatingBelow should match the set value");
    }
}
