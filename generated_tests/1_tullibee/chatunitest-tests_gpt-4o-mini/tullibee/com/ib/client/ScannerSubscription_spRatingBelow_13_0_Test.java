package com.ib.client;

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
    public void testSpRatingBelow() {
        // Test default value
        assertEquals(null, scannerSubscription.spRatingBelow());
        // Test setting a value
        String expectedRating = "A";
        scannerSubscription.spRatingAbove(expectedRating);
        assertEquals(expectedRating, scannerSubscription.spRatingBelow());
        // Test setting another value
        String anotherExpectedRating = "B";
        scannerSubscription.spRatingAbove(anotherExpectedRating);
        assertEquals(anotherExpectedRating, scannerSubscription.spRatingBelow());
    }
}
