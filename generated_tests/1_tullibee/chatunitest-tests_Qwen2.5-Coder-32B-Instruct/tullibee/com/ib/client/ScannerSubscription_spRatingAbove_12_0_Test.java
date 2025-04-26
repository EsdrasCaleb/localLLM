package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_12_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testSpRatingAbove_DefaultValue() throws NoSuchFieldException, IllegalAccessException {
        // Access private field m_spRatingAbove using reflection
        Field spRatingAboveField = ScannerSubscription.class.getDeclaredField("m_spRatingAbove");
        spRatingAboveField.setAccessible(true);
        // Ensure the default value is null
        assertEquals(null, spRatingAboveField.get(scannerSubscription));
        // Verify that spRatingAbove() returns the default value
        assertEquals(null, scannerSubscription.spRatingAbove());
    }

    @Test
    public void testSpRatingAbove_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Set a value using the setter method
        scannerSubscription.spRatingAbove("AAA");
        // Access private field m_spRatingAbove using reflection to verify
        Field spRatingAboveField = ScannerSubscription.class.getDeclaredField("m_spRatingAbove");
        spRatingAboveField.setAccessible(true);
        // Ensure the value is set correctly
        assertEquals("AAA", spRatingAboveField.get(scannerSubscription));
        // Verify that spRatingAbove() returns the set value
        assertEquals("AAA", scannerSubscription.spRatingAbove());
    }
}
