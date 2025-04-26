package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_11_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingBelow_WhenNotSet_ShouldReturnNull() {
        assertEquals(null, scannerSubscription.moodyRatingBelow());
    }

    @Test
    public void testMoodyRatingBelow_WhenSetToValue_ShouldReturnValue() {
        String expectedRating = "Baa3";
        scannerSubscription.moodyRatingBelow(expectedRating);
        assertEquals(expectedRating, scannerSubscription.moodyRatingBelow());
    }

    @Test
    public void testMoodyRatingBelow_WhenSetToEmptyString_ShouldReturnEmptyString() {
        scannerSubscription.moodyRatingBelow("");
        assertEquals("", scannerSubscription.moodyRatingBelow());
    }

    @Test
    public void testMoodyRatingBelow_WhenSetToNull_ShouldReturnNull() {
        scannerSubscription.moodyRatingBelow(null);
        assertEquals(null, scannerSubscription.moodyRatingBelow());
    }
}
