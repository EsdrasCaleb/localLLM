package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_10_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingAbove() {
        // Test with null value
        scannerSubscription.moodyRatingAbove(null);
        assertEquals(null, scannerSubscription.moodyRatingAbove());
        // Test with an empty string
        scannerSubscription.moodyRatingAbove("");
        assertEquals("", scannerSubscription.moodyRatingAbove());
        // Test with a valid string
        String expectedRating = "Aaa";
        scannerSubscription.moodyRatingAbove(expectedRating);
        assertEquals(expectedRating, scannerSubscription.moodyRatingAbove());
    }
}
